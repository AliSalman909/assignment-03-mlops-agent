"""Tests for the Airflow workflow module (Task 3) using httpx.MockTransport (no server needed)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from agent.config import Settings
from agent.schemas import ToolStatus
from agent.tools.airflow_tools import AirflowClient, AirflowTools, register_airflow_tools
from agent.tools.base import ToolRegistry

DAG = "churn_training_pipeline"
SECRET = "s3cr3t-airflow-pw"
TASKS = ["validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate", "register_candidate"]


class FakeAirflow:
    """Minimal Airflow 3 API: token auth, dags, runs, task instances, logs."""

    def __init__(self, scenario: str = "success"):
        self.scenario = scenario
        self.calls: list[str] = []
        self.tokens_issued = 0
        self.expire_next = False
        self.polls = 0

    def task_states(self):
        if self.scenario == "corrupt":
            return {"validate_dataset": ("failed", 1)} | {t: ("upstream_failed", 0) for t in TASKS[1:]}
        if self.scenario == "retry":
            return {t: ("success", 2 if t == "preprocess_data" else 1) for t in TASKS}
        return {t: ("success", 1) for t in TASKS}

    def run(self, state=None):
        st = state or {"success": "success", "corrupt": "failed", "retry": "success"}[self.scenario]
        return {
            "dag_run_id": "manual__1",
            "state": st,
            "run_type": "manual",
            "start_date": "2026-11-15T10:00:00Z",
            "end_date": "2026-11-15T10:02:00Z",
            "conf": {},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        if path == "/auth/token":
            body = json.loads(request.content)
            if body["password"] != SECRET:
                return httpx.Response(401, json={"detail": "bad creds"})
            self.tokens_issued += 1
            return httpx.Response(201, json={"access_token": f"tok{self.tokens_issued}"})
        if not request.headers.get("Authorization", "").startswith("Bearer tok"):
            return httpx.Response(401)
        if self.expire_next:
            self.expire_next = False
            return httpx.Response(401)
        if path == "/api/v2/dags":
            return httpx.Response(
                200,
                json={
                    "dags": [
                        {"dag_id": DAG, "is_paused": False, "tags": [{"name": "ml"}]},
                        {"dag_id": "etl_other", "is_paused": False, "tags": []},
                    ]
                },
            )
        if path == f"/api/v2/dags/{DAG}":
            return httpx.Response(200, json={"dag_id": DAG, "is_paused": False})
        if path == f"/api/v2/dags/{DAG}/dagRuns" and request.method == "POST":
            return httpx.Response(200, json=self.run("queued"))
        if path == f"/api/v2/dags/{DAG}/dagRuns":
            return httpx.Response(200, json={"dag_runs": [self.run()], "total_entries": 1})
        if path.endswith("/dagRuns/manual__1"):
            self.polls += 1
            return httpx.Response(200, json=self.run("running" if self.polls < 2 else None))
        if path.endswith("/taskInstances"):
            tis = [
                {"task_id": t, "state": s, "try_number": n, "max_tries": 2, "start_date": f"2026-11-15T10:0{i}:00Z"}
                for i, (t, (s, n)) in enumerate(self.task_states().items())
            ]
            return httpx.Response(200, json={"task_instances": tis})
        if "/logs/" in path:
            task = path.split("/taskInstances/")[1].split("/")[0]
            if task == "validate_dataset" and self.scenario == "corrupt":
                content = [
                    {"timestamp": "t", "level": "warning", "event": "INJECTED FAILURE: label column lost"},
                    {
                        "timestamp": "t",
                        "level": "error",
                        "event": "Task failed with exception",
                        "error_detail": [
                            {
                                "exc_type": "AirflowFailException",
                                "exc_value": "DATA_VALIDATION_FAILED: required_columns: missing=['churned']",
                            }
                        ],
                    },
                ]
            elif task == "preprocess_data" and self.scenario == "retry":
                content = [{"timestamp": "t", "level": "error", "event": "TRANSIENT: object storage read timed out"}]
            else:
                content = [{"timestamp": "t", "level": "info", "event": f"{task} done password={SECRET}"}]
            return httpx.Response(200, json={"content": content})
        return httpx.Response(404)


def _registry(fake: FakeAirflow, password: str = SECRET) -> ToolRegistry:
    settings = Settings(airflow_password=password, airflow_approved_dags=(DAG,))
    client = AirflowClient(settings, transport=httpx.MockTransport(fake))
    reg = ToolRegistry()
    register_airflow_tools(reg, AirflowTools(settings=settings, client=client, sleep=lambda s: None))
    return reg


def test_list_dags_authenticates_and_marks_approved():
    fake = FakeAirflow()
    res = _registry(fake).execute("list_dags", {})
    assert res.ok, res.error
    assert fake.calls[0] == "POST /auth/token"
    approved = {d["dag_id"]: d["approved_for_agent_trigger"] for d in res.data["dags"]}
    assert approved == {DAG: True, "etl_other": False}


def test_trigger_requires_confirmation_and_makes_no_api_call():
    fake = FakeAirflow()
    res = _registry(fake).execute("trigger_training_pipeline", {"dag_id": DAG})
    assert res.status == ToolStatus.CONFIRMATION_REQUIRED
    assert fake.calls == []  # nothing reached Airflow


def test_trigger_rejects_unapproved_dag_and_unknown_conf():
    fake = FakeAirflow()
    reg = _registry(fake)
    assert (
        reg.execute("trigger_training_pipeline", {"dag_id": "etl_other"}, confirmed=True).status == ToolStatus.FORBIDDEN
    )
    bad = reg.execute("trigger_training_pipeline", {"config": {"rm_rf": True}}, confirmed=True)
    assert bad.status == ToolStatus.INVALID_REQUEST


def test_confirmed_trigger_waits_and_reports_success():
    res = _registry(FakeAirflow()).execute("trigger_training_pipeline", {"wait_seconds": 60}, confirmed=True)
    assert res.ok, res.error
    assert res.data["state"] == "success" and res.data["finished_within_wait"] is True
    assert res.data["task_counts"] == {"success": 6}


def test_diagnose_data_validation_failure_blocks_promotion():
    reg = _registry(FakeAirflow("corrupt"))
    failed = reg.execute("get_failed_tasks", {})
    assert [t["task_id"] for t in failed.data["failed_tasks"]] == ["validate_dataset"]
    res = reg.execute("diagnose_dag_run", {})
    assert res.ok, res.error
    f = res.data["findings"][0]
    assert f["task_id"] == "validate_dataset" and f["failure_layer"] == "data"
    assert any("DATA_VALIDATION_FAILED" in e for e in f["evidence"])
    assert res.data["promotion_blocked"] is True


def test_diagnose_detects_recovered_transient_failure():
    res = _registry(FakeAirflow("retry")).execute("diagnose_dag_run", {})
    assert res.ok
    assert res.data["findings"][0]["failure_layer"] == "infrastructure (recovered)"
    assert res.data["diagnosis"] == "run succeeded after retries"


def test_expired_token_is_refreshed_once():
    fake = FakeAirflow()
    reg = _registry(fake)
    reg.execute("list_dags", {})
    fake.expire_next = True
    assert reg.execute("list_dags", {}).ok
    assert fake.tokens_issued == 2


def test_bad_credentials_are_forbidden_and_not_leaked(caplog):
    caplog.set_level(logging.INFO, logger="mlops_agent")
    res = _registry(FakeAirflow(), password="wrong-password").execute("list_dags", {})
    assert res.status == ToolStatus.FORBIDDEN
    assert "wrong-password" not in caplog.text and "wrong-password" not in (res.error or "")


def test_task_logs_redact_secrets():
    res = _registry(FakeAirflow()).execute("get_task_logs", {"task_id": "train_model"})
    assert res.ok
    assert SECRET not in json.dumps(res.data)


@pytest.mark.parametrize(
    "exc, status",
    [(httpx.ConnectError("refused"), ToolStatus.UNAVAILABLE), (httpx.ReadTimeout("slow"), ToolStatus.TIMEOUT)],
)
def test_airflow_unreachable_or_slow(exc, status):
    def handler(request):
        raise exc

    settings = Settings(airflow_password=SECRET)
    client = AirflowClient(settings, transport=httpx.MockTransport(handler))
    reg = ToolRegistry()
    register_airflow_tools(reg, AirflowTools(settings=settings, client=client))
    res = reg.execute("get_dag_runs", {})
    assert res.status == status and res.data is None


def test_trigger_reports_run_even_if_polling_fails():
    class FlakyPolling(FakeAirflow):
        def __call__(self, request):
            if request.url.path.endswith("/dagRuns/manual__1"):
                return httpx.Response(503)
            return super().__call__(request)

    res = _registry(FlakyPolling()).execute("trigger_training_pipeline", {"wait_seconds": 30}, confirmed=True)
    assert res.ok and res.data["triggered"] is True and res.data["run_id"] == "manual__1"
    assert res.data["finished_within_wait"] is False and "do not re-trigger" in res.data["next_step"]
