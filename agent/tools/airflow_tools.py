"""Task 3 - Apache Airflow Workflow module (Airflow 3 REST API v2, JWT-authenticated).

Tools
  list_dags            DAGs with paused state, tags and whether the agent may trigger them
  get_dag_runs         recent runs of a DAG with state, timing and conf
  get_dag_run_status   one run (default: latest) with per-task states; can wait for completion
  trigger_training_pipeline  STATE-CHANGING: start an approved DAG (requires confirmation + allow-list)
  get_failed_tasks     failed / upstream_failed tasks of a run, with attempt counts
  get_task_logs        log tail of a task attempt (redacted)
  diagnose_dag_run     root-cause analysis: failing task, error excerpt, failure layer, retries, next steps

Auth: POST /auth/token with username/password from the environment -> Bearer token, cached and
refreshed once on 401. Credentials are never logged or returned.
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, field_validator

from agent.config import Settings, get_settings
from agent.logging_utils import redact
from agent.tools.base import (
    Tool,
    ToolError,
    ToolForbiddenError,
    ToolInvalidRequestError,
    ToolOutput,
    ToolRegistry,
    ToolTimeoutError,
    ToolUnavailableError,
)

MODULE = "airflow"
TERMINAL_STATES = {"success", "failed"}
_ID_RE = re.compile(r"^[A-Za-z0-9_.:+\-]{1,250}$")
ALLOWED_CONF_KEYS = {"dataset_path", "model_type", "min_f1", "min_roc_auc", "inject_failure"}

# error signatures -> (layer, explanation, recommended action)
FAILURE_SIGNATURES: list[tuple[str, str, str, str]] = [
    (
        "DATA_VALIDATION_FAILED",
        "data",
        "the input dataset failed validation checks",
        "inspect the dataset version with the DVC tools; fix or roll back the data, then re-run",
    ),
    (
        "QUALITY_GATE_FAILED",
        "model",
        "the trained model did not meet the quality thresholds",
        "compare against previous runs in MLflow; do not promote; tune features/hyper-parameters or check data drift",
    ),
    (
        "TRANSIENT",
        "infrastructure",
        "a transient infrastructure error occurred",
        "usually self-heals via retries; if persistent, check storage/network health",
    ),
    (
        "ModuleNotFoundError",
        "environment",
        "a Python dependency is missing in the worker image",
        "add the package to the Airflow image and rebuild",
    ),
    (
        "ImportError",
        "environment",
        "a Python import failed in the worker",
        "check package versions in the Airflow image",
    ),
    ("MemoryError", "infrastructure", "the task ran out of memory", "increase worker memory or reduce data size"),
    (
        "Connection refused",
        "infrastructure",
        "a dependent service was unreachable",
        "check that MLflow / storage services are running",
    ),
    ("TimeoutError", "infrastructure", "an operation timed out", "check service latency; retries may succeed"),
]


# ---------------------------------------------------------------------------- inputs
def _check_id(v: str, what: str) -> str:
    v = v.strip()
    if not _ID_RE.match(v):
        raise ValueError(f"invalid {what} '{v}'")
    return v


class ListDagsInput(BaseModel):
    limit: int = Field(100, ge=1, le=500)


class DagRunsInput(BaseModel):
    dag_id: str = Field("churn_training_pipeline")
    limit: int = Field(5, ge=1, le=50)

    @field_validator("dag_id")
    @classmethod
    def v_dag(cls, v: str) -> str:
        return _check_id(v, "dag_id")


class DagRunRefInput(BaseModel):
    dag_id: str = Field("churn_training_pipeline")
    run_id: str | None = Field(None, description="DAG run id; omit for the most recent run")

    @field_validator("dag_id")
    @classmethod
    def v_dag(cls, v: str) -> str:
        return _check_id(v, "dag_id")

    @field_validator("run_id")
    @classmethod
    def v_run(cls, v: str | None) -> str | None:
        return v if v is None else _check_id(v, "run_id")


class DagRunStatusInput(DagRunRefInput):
    wait_seconds: int = Field(0, ge=0, le=900, description="Poll until the run finishes, up to this many seconds")


class TriggerDagInput(BaseModel):
    dag_id: str = Field("churn_training_pipeline")
    config: dict[str, Any] = Field(
        default_factory=dict, description=f"Run parameters; allowed keys: {sorted(ALLOWED_CONF_KEYS)}"
    )
    wait_seconds: int = Field(0, ge=0, le=900, description="Wait for the run to finish and report the outcome")

    @field_validator("dag_id")
    @classmethod
    def v_dag(cls, v: str) -> str:
        return _check_id(v, "dag_id")

    @field_validator("config")
    @classmethod
    def v_conf(cls, v: dict[str, Any]) -> dict[str, Any]:
        unknown = set(v) - ALLOWED_CONF_KEYS
        if unknown:
            raise ValueError(f"config keys not allowed: {sorted(unknown)}")
        for k, val in v.items():
            if not isinstance(val, str | int | float | bool) or (isinstance(val, str) and len(val) > 200):
                raise ValueError(f"config value for '{k}' must be a short scalar")
        return v


class TaskLogsInput(DagRunRefInput):
    task_id: str
    try_number: int | None = Field(None, ge=1, le=20, description="Attempt number; omit for the latest attempt")
    max_lines: int = Field(60, ge=5, le=400)

    @field_validator("task_id")
    @classmethod
    def v_task(cls, v: str) -> str:
        return _check_id(v, "task_id")


# ---------------------------------------------------------------------------- client
class AirflowClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._http = httpx.Client(
            base_url=settings.airflow_base_url.rstrip("/"), timeout=settings.airflow_timeout_s, transport=transport
        )
        self._token: str | None = None

    def _authenticate(self) -> None:
        if not self.settings.airflow_password:
            raise ToolUnavailableError("Airflow credentials are not configured (AIRFLOW_PASSWORD)")
        resp = self._send(
            "POST",
            "/auth/token",
            json={"username": self.settings.airflow_username, "password": self.settings.airflow_password},
            auth=False,
        )
        if resp.status_code in (401, 403):
            raise ToolForbiddenError("Airflow rejected the configured credentials")
        if resp.status_code >= 400:
            raise ToolError(f"Airflow token endpoint returned HTTP {resp.status_code}")
        self._token = resp.json()["access_token"]

    def _send(self, method: str, path: str, *, auth: bool = True, **kw) -> httpx.Response:
        headers = kw.pop("headers", {})
        if auth:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            return self._http.request(method, path, headers=headers, **kw)
        except httpx.TimeoutException as exc:
            raise ToolTimeoutError(f"Airflow API timed out after {self.settings.airflow_timeout_s:.0f}s") from exc
        except httpx.TransportError as exc:
            raise ToolUnavailableError(f"Airflow API at {self.settings.airflow_base_url} is unreachable") from exc

    def request(self, method: str, path: str, **kw) -> Any:
        if self._token is None:
            self._authenticate()
        resp = self._send(method, path, **kw)
        if resp.status_code == 401:  # token expired -> refresh once
            self._authenticate()
            resp = self._send(method, path, **kw)
        if resp.status_code == 404:
            raise ToolInvalidRequestError(f"Airflow: not found ({path})")
        if resp.status_code == 403:
            raise ToolForbiddenError(f"Airflow denied access to {path}")
        if resp.status_code in (400, 409, 422):
            raise ToolInvalidRequestError(f"Airflow rejected the request: {redact(resp.text[:300])}")
        if resp.status_code >= 500:
            raise ToolUnavailableError(f"Airflow API error HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise ToolError(f"Airflow API HTTP {resp.status_code}: {redact(resp.text[:200])}")
        return resp.json() if resp.content else {}


# ---------------------------------------------------------------------------- tools
class AirflowTools:
    def __init__(self, settings: Settings | None = None, client: AirflowClient | None = None, sleep=time.sleep) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._sleep = sleep

    @property
    def api(self) -> AirflowClient:
        if self._client is None:
            self._client = AirflowClient(self.settings)
        return self._client

    # ----- helpers
    @staticmethod
    def _run_summary(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": r.get("dag_run_id"),
            "state": r.get("state"),
            "run_type": r.get("run_type"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
            "duration_s": r.get("duration"),
            "conf": r.get("conf"),
            "note": r.get("note"),
        }

    def _runs(self, dag_id: str, limit: int) -> list[dict[str, Any]]:
        data = self.api.request(
            "GET", f"/api/v2/dags/{quote(dag_id)}/dagRuns", params={"limit": limit, "order_by": "-run_after"}
        )
        return data.get("dag_runs", [])

    def _resolve_run(self, dag_id: str, dag_run_id: str | None) -> dict[str, Any]:
        if dag_run_id:
            return self.api.request("GET", f"/api/v2/dags/{quote(dag_id)}/dagRuns/{quote(dag_run_id, safe='')}")
        runs = self._runs(dag_id, 1)
        if not runs:
            raise ToolInvalidRequestError(f"DAG '{dag_id}' has no runs yet")
        return runs[0]

    def _tasks(self, dag_id: str, run_id: str) -> list[dict[str, Any]]:
        data = self.api.request(
            "GET",
            f"/api/v2/dags/{quote(dag_id)}/dagRuns/{quote(run_id, safe='')}/taskInstances",
            params={"limit": 100},
        )
        tis = data.get("task_instances", [])
        return [
            {
                "task_id": t.get("task_id"),
                "state": t.get("state"),
                "try_number": t.get("try_number"),
                "max_tries": t.get("max_tries"),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date"),
                "duration_s": t.get("duration"),
                "operator": t.get("operator"),
            }
            for t in sorted(tis, key=lambda t: t.get("start_date") or "9999")
        ]

    def _logs(self, dag_id: str, run_id: str, task_id: str, try_number: int) -> list[str]:
        data = self.api.request(
            "GET",
            f"/api/v2/dags/{quote(dag_id)}/dagRuns/{quote(run_id, safe='')}/taskInstances/{quote(task_id)}"
            f"/logs/{try_number}",
            params={"full_content": "true"},
            headers={"Accept": "application/json"},
        )
        content = data.get("content", "")
        lines: list[str] = []
        if isinstance(content, str):
            lines = content.splitlines()
        else:
            for item in content:
                if isinstance(item, dict):
                    level = (item.get("level") or "").upper()
                    lines.append(f"{item.get('timestamp', '')} {level} {item.get('event', '')}".strip())
                    for err in item.get("error_detail") or []:
                        if isinstance(err, dict):
                            lines.append(f"    {err.get('exc_type')}: {err.get('exc_value')}")
                else:
                    lines.append(str(item))
        # drop Airflow UI grouping markers (::group:: / ::endgroup::) - noise for diagnosis
        return [redact(ln) for ln in lines if ln.strip() and "::group::" not in ln and "::endgroup::" not in ln]

    def _wait(self, dag_id: str, run_id: str, wait_seconds: int) -> tuple[dict[str, Any], bool]:
        deadline = time.monotonic() + wait_seconds
        run = self._resolve_run(dag_id, run_id)
        while run.get("state") not in TERMINAL_STATES and time.monotonic() < deadline:
            self._sleep(5)
            run = self._resolve_run(dag_id, run_id)
        return run, run.get("state") in TERMINAL_STATES

    def _status_payload(self, dag_id: str, run: dict[str, Any], finished: bool | None = None) -> dict[str, Any]:
        tasks = self._tasks(dag_id, run["dag_run_id"])
        counts: dict[str, int] = {}
        for t in tasks:
            counts[t["state"] or "none"] = counts.get(t["state"] or "none", 0) + 1
        payload = {"dag_id": dag_id, **self._run_summary(run), "task_counts": counts, "tasks": tasks}
        if finished is not None:
            payload["finished_within_wait"] = finished
        retried = [t["task_id"] for t in tasks if (t["try_number"] or 0) > 1]
        if retried:
            payload["tasks_retried"] = retried
        return payload

    # ----- tools
    def list_dags(self, p: ListDagsInput) -> ToolOutput:
        data = self.api.request("GET", "/api/v2/dags", params={"limit": p.limit})
        dags = [
            {
                "dag_id": d.get("dag_id"),
                "description": d.get("description"),
                "is_paused": d.get("is_paused"),
                "tags": [t.get("name") if isinstance(t, dict) else t for t in d.get("tags") or []],
                "schedule": d.get("timetable_summary") or d.get("timetable_description"),
                "approved_for_agent_trigger": d.get("dag_id") in self.settings.airflow_approved_dags,
            }
            for d in data.get("dags", [])
        ]
        return ToolOutput(data={"count": len(dags), "dags": dags}, evidence_source=["GET /api/v2/dags"])

    def get_dag_runs(self, p: DagRunsInput) -> ToolOutput:
        runs = [self._run_summary(r) for r in self._runs(p.dag_id, p.limit)]
        states: dict[str, int] = {}
        for r in runs:
            states[r["state"]] = states.get(r["state"], 0) + 1
        return ToolOutput(
            data={"dag_id": p.dag_id, "count": len(runs), "state_counts": states, "runs": runs},
            evidence_source=[f"GET /api/v2/dags/{p.dag_id}/dagRuns"],
        )

    def get_dag_run_status(self, p: DagRunStatusInput) -> ToolOutput:
        run = self._resolve_run(p.dag_id, p.run_id)
        finished = None
        if p.wait_seconds:
            run, finished = self._wait(p.dag_id, run["dag_run_id"], p.wait_seconds)
        return ToolOutput(
            data=self._status_payload(p.dag_id, run, finished),
            evidence_source=[f"GET dagRuns/{run['dag_run_id']}", "GET taskInstances"],
        )

    def trigger_training_pipeline(self, p: TriggerDagInput) -> ToolOutput:
        if p.dag_id not in self.settings.airflow_approved_dags:
            raise ToolForbiddenError(
                f"DAG '{p.dag_id}' is not on the approved list {list(self.settings.airflow_approved_dags)}"
            )
        dag = self.api.request("GET", f"/api/v2/dags/{quote(p.dag_id)}")
        if dag.get("is_paused"):
            raise ToolInvalidRequestError(f"DAG '{p.dag_id}' is paused; an operator must unpause it first")
        run = self.api.request(
            "POST",
            f"/api/v2/dags/{quote(p.dag_id)}/dagRuns",
            json={"logical_date": None, "conf": p.config, "note": "triggered by MLOps agent after user confirmation"},
        )
        sources = [f"POST /api/v2/dags/{p.dag_id}/dagRuns"]
        data: dict[str, Any] = {"dag_id": p.dag_id, **self._run_summary(run)}
        if p.wait_seconds:
            # The run now exists. A polling failure must not turn into "trigger failed" - that would invite
            # a second, duplicate training run. Report the run id and that its outcome is not yet known.
            try:
                polled, finished = self._wait(p.dag_id, run["dag_run_id"], p.wait_seconds)
                data = self._status_payload(p.dag_id, polled, finished)
                sources.append("polled run + task instance states")
            except ToolError as exc:
                data["finished_within_wait"] = False
                data["polling_error"] = f"run was triggered but its status could not be read: {exc}"
                data["next_step"] = (
                    f"check later with get_dag_run_status(run_id='{run['dag_run_id']}'); do not re-trigger"
                )
        data["triggered"] = True
        return ToolOutput(data=data, evidence_source=sources)

    def get_failed_tasks(self, p: DagRunRefInput) -> ToolOutput:
        run = self._resolve_run(p.dag_id, p.run_id)
        tasks = self._tasks(p.dag_id, run["dag_run_id"])
        failed = [t for t in tasks if t["state"] == "failed"]
        blocked = [t for t in tasks if t["state"] in ("upstream_failed", "skipped")]
        return ToolOutput(
            data={
                "dag_id": p.dag_id,
                "run_id": run["dag_run_id"],
                "run_state": run.get("state"),
                "failed_tasks": failed,
                "blocked_downstream_tasks": blocked,
                "summary": (
                    f"{len(failed)} task(s) failed; {len(blocked)} downstream task(s) did not run"
                    if failed
                    else "no failed tasks in this run"
                ),
            },
            evidence_source=[f"GET dagRuns/{run['dag_run_id']}/taskInstances"],
        )

    def get_task_logs(self, p: TaskLogsInput) -> ToolOutput:
        run = self._resolve_run(p.dag_id, p.run_id)
        try_number = p.try_number
        if try_number is None:
            task = next((t for t in self._tasks(p.dag_id, run["dag_run_id"]) if t["task_id"] == p.task_id), None)
            if task is None:
                raise ToolInvalidRequestError(f"task '{p.task_id}' not found in run {run['dag_run_id']}")
            try_number = max(task["try_number"] or 1, 1)
        lines = self._logs(p.dag_id, run["dag_run_id"], p.task_id, try_number)
        return ToolOutput(
            data={
                "dag_id": p.dag_id,
                "run_id": run["dag_run_id"],
                "task_id": p.task_id,
                "try_number": try_number,
                "total_lines": len(lines),
                "lines": lines[-p.max_lines :],
            },
            evidence_source=[f"GET taskInstances/{p.task_id}/logs/{try_number}"],
        )

    def diagnose_dag_run(self, p: DagRunRefInput) -> ToolOutput:
        run = self._resolve_run(p.dag_id, p.run_id)
        run_id = run["dag_run_id"]
        tasks = self._tasks(p.dag_id, run_id)
        failed = [t for t in tasks if t["state"] == "failed"]
        retried_ok = [t for t in tasks if (t["try_number"] or 0) > 1 and t["state"] == "success"]
        data: dict[str, Any] = {
            "dag_id": p.dag_id,
            "run_id": run_id,
            "run_state": run.get("state"),
            "tasks": [{k: t[k] for k in ("task_id", "state", "try_number")} for t in tasks],
        }
        sources = [f"GET dagRuns/{run_id}", "GET taskInstances"]

        findings = []
        for t in failed:
            lines = self._logs(p.dag_id, run_id, t["task_id"], max(t["try_number"] or 1, 1))
            sources.append(f"GET logs {t['task_id']}#{t['try_number']}")
            joined = "\n".join(lines)
            match = next((s for s in FAILURE_SIGNATURES if s[0] in joined), None)
            error_lines = [ln for ln in lines if re.search(r"ERROR|Exception|Error:|FAILED|INJECTED", ln)][-8:]
            findings.append(
                {
                    "task_id": t["task_id"],
                    "attempts": t["try_number"],
                    "failure_layer": match[1] if match else "unknown",
                    "explanation": match[2] if match else "no known failure signature found in logs",
                    "recommended_action": match[3] if match else "read the full task log",
                    "evidence": error_lines or lines[-8:],
                }
            )
        for t in retried_ok:
            first = self._logs(p.dag_id, run_id, t["task_id"], 1)
            sources.append(f"GET logs {t['task_id']}#1")
            findings.append(
                {
                    "task_id": t["task_id"],
                    "attempts": t["try_number"],
                    "failure_layer": "infrastructure (recovered)",
                    "explanation": "first attempt failed, a retry succeeded - retry policy absorbed a transient fault",
                    "recommended_action": "no action unless this recurs",
                    "evidence": [ln for ln in first if re.search(r"ERROR|Error|TRANSIENT|INJECTED", ln)][-5:],
                }
            )
        blocked = [t["task_id"] for t in tasks if t["state"] == "upstream_failed"]
        data["findings"] = findings
        data["blocked_downstream_tasks"] = blocked
        if "register_candidate" in blocked:
            data["promotion_blocked"] = True
        if run.get("state") == "success" and not findings:
            data["diagnosis"] = "run succeeded with no failures or retries"
        elif failed:
            root = findings[0]
            data["diagnosis"] = (
                f"root cause in task '{root['task_id']}' ({root['failure_layer']} layer): {root['explanation']}"
            )
        elif run.get("state") not in TERMINAL_STATES:
            data["diagnosis"] = f"run is still {run.get('state')}; no failures so far"
        else:
            data["diagnosis"] = "run succeeded after retries" if retried_ok else f"run state is {run.get('state')}"
        return ToolOutput(data=data, evidence_source=sources)


def register_airflow_tools(registry: ToolRegistry, tools: AirflowTools | None = None) -> AirflowTools:
    t = tools or AirflowTools()
    specs = [
        (
            "list_dags",
            ListDagsInput,
            t.list_dags,
            False,
            "List Airflow DAGs with paused state, tags and whether the agent is approved to trigger them.",
        ),
        (
            "get_dag_runs",
            DagRunsInput,
            t.get_dag_runs,
            False,
            "List the most recent runs of a DAG (default churn_training_pipeline) with state, timing and conf.",
        ),
        (
            "get_dag_run_status",
            DagRunStatusInput,
            t.get_dag_run_status,
            False,
            "Status of one DAG run (default: latest) including every task's state and attempts. Set wait_seconds "
            "to poll until it finishes.",
        ),
        (
            "trigger_training_pipeline",
            TriggerDagInput,
            t.trigger_training_pipeline,
            True,
            "STATE-CHANGING: trigger an approved DAG (e.g. the training pipeline). Requires explicit user "
            "confirmation; never call it just because retraining might help. Set wait_seconds to report the outcome.",
        ),
        (
            "get_failed_tasks",
            DagRunRefInput,
            t.get_failed_tasks,
            False,
            "List failed tasks and the downstream tasks they blocked for a DAG run (default: latest).",
        ),
        (
            "get_task_logs",
            TaskLogsInput,
            t.get_task_logs,
            False,
            "Fetch the tail of a task attempt's log (secrets redacted).",
        ),
        (
            "diagnose_dag_run",
            DagRunRefInput,
            t.diagnose_dag_run,
            False,
            "Diagnose a DAG run: root failing task, error excerpt from logs, failure layer (data/model/"
            "infrastructure/environment), retries, whether promotion was blocked, and recommended action.",
        ),
    ]
    for name, model, func, mutating, desc in specs:
        registry.register(
            Tool(
                name=name,
                module=MODULE,
                description=desc,
                input_model=model,
                func=func,
                mutating=mutating,
                requires_confirmation=mutating,
            )
        )
    return t
