"""Agent-loop tests with a scripted LLM (deterministic, no Gemini calls)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from agent.config import Settings
from agent.llm import FunctionCall, LLMError, LLMTurn
from agent.orchestrator import MLOpsAgent, required_modules
from agent.schemas import ToolStatus
from agent.tools.base import Tool, ToolOutput, ToolRegistry, ToolUnavailableError


class NoArgs(BaseModel):
    pass


class RunArgs(BaseModel):
    run_id: str


class ScriptedLLM:
    """Returns pre-programmed turns and records everything it was shown."""

    model_name = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def generate(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages])
        if not self.turns:
            return LLMTurn(text="done")
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        return t


def calls(*pairs):
    return LLMTurn(text=None, calls=[FunctionCall(name=n, args=a, id=f"c{i}") for i, (n, a) in enumerate(pairs)])


def answer(text):
    return LLMTurn(text=text)


@pytest.fixture()
def registry():
    executed: list[str] = []
    reg = ToolRegistry()

    def ok(name, data):
        def f(p):
            executed.append(name)
            return ToolOutput(data=data, evidence_source=[name])

        return f

    def down(p):
        executed.append("get_deployed_version")
        raise ToolUnavailableError("Kubernetes API server is unreachable")

    reg.register(
        Tool("find_best_model", "mlflow", "best run", NoArgs, ok("find_best_model", {"run_id": "r1", "f1": 0.51}))
    )
    reg.register(Tool("get_run_details", "mlflow", "run details", RunArgs, ok("get_run_details", {"f1": 0.51})))
    reg.register(Tool("get_dataset_version", "dvc", "dataset", NoArgs, ok("get_dataset_version", {"version": "v2.0"})))
    reg.register(Tool("get_dag_runs", "airflow", "runs", NoArgs, ok("get_dag_runs", {"runs": []})))
    reg.register(Tool("get_retraining_policy", "policy", "policy", NoArgs, ok("get_retraining_policy", {"rules": []})))
    reg.register(Tool("get_deployed_version", "kubernetes", "version", NoArgs, down))
    reg.register(
        Tool(
            "trigger_training_pipeline",
            "airflow",
            "trigger",
            NoArgs,
            ok("trigger_training_pipeline", {"triggered": True}),
            mutating=True,
            requires_confirmation=True,
        )
    )
    reg.executed = executed
    return reg


def make_agent(registry, turns, **kw):
    llm = ScriptedLLM(turns)
    return MLOpsAgent(registry, llm, Settings(**kw)), llm


def test_multi_tool_request_executes_selected_tools(registry):
    agent, llm = make_agent(
        registry,
        [
            calls(("find_best_model", {}), ("get_dataset_version", {})),
            calls(("get_run_details", {"run_id": "r1"})),
            answer("## Known facts\n- best run r1 trained on v2.0"),
        ],
    )
    resp = agent.run("Which dataset trained the best model?")
    assert [c.tool for c in resp.tool_calls] == ["find_best_model", "get_dataset_version", "get_run_details"]
    assert all(c.status == ToolStatus.OK for c in resp.tool_calls)
    assert registry.executed == ["find_best_model", "get_dataset_version", "get_run_details"]
    assert resp.answer.startswith("## Known facts") and resp.steps == 3
    # the LLM received structured results with status + data for each call
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["results"][0]["response"]["status"] == "ok"


def test_unavailable_tool_is_reported_not_invented(registry):
    agent, llm = make_agent(
        registry, [calls(("get_deployed_version", {})), answer("The deployment is running version 9.9 and is healthy.")]
    )
    resp = agent.run("What version is deployed?")
    payload = llm.seen[1][-1]["results"][0]["response"]
    assert payload["status"] == "unavailable" and payload["data"] is None and "UNKNOWN" in payload["note"]
    assert resp.data_gaps == ["get_deployed_version: unavailable - Kubernetes API server is unreachable"]
    # even if the LLM hallucinates, the response carries a deterministic, visible data-gap disclosure
    assert "Data gaps" in resp.answer and "unavailable" in resp.answer


def test_retraining_question_never_triggers_training(registry):
    agent, _ = make_agent(registry, [calls(("trigger_training_pipeline", {})), answer("I queued retraining.")])
    resp = agent.run("Is retraining necessary?")
    assert "trigger_training_pipeline" not in registry.executed  # never ran
    assert resp.tool_calls[0].status == ToolStatus.CONFIRMATION_REQUIRED
    assert len(resp.pending_actions) == 1 and "Awaiting your confirmation" in resp.answer


def test_confirmation_runs_exact_action_once(registry):
    agent, _ = make_agent(
        registry,
        [calls(("trigger_training_pipeline", {})), answer("awaiting confirmation"), answer("- pipeline triggered")],
    )
    action = agent.run("Trigger the training pipeline").pending_actions[0]
    first = agent.confirm(action.action_id)
    assert first.tool_calls[0].status == ToolStatus.OK and registry.executed == ["trigger_training_pipeline"]
    second = agent.confirm(action.action_id)
    assert "Nothing was executed" in second.answer and registry.executed == ["trigger_training_pipeline"]


def test_invalid_tool_requests_are_rejected_gracefully(registry):
    agent, _ = make_agent(
        registry, [calls(("delete_everything", {}), ("get_run_details", {"run_id": 7})), answer("could not do that")]
    )
    resp = agent.run("delete everything")
    assert [c.status for c in resp.tool_calls] == [ToolStatus.INVALID_REQUEST, ToolStatus.INVALID_REQUEST]
    assert registry.executed == []


def test_llm_failure_returns_honest_answer(registry):
    agent, _ = make_agent(registry, [calls(("find_best_model", {})), LLMError("Gemini request failed: timeout")])
    resp = agent.run("best model?")
    assert "language model is unavailable" in resp.answer and "No conclusions" in resp.answer


def test_step_budget_and_duplicate_call_cache(registry):
    loop = [calls(("find_best_model", {}))] * 5
    agent, _ = make_agent(registry, loop, agent_max_steps=3)
    resp = agent.run("best model?")
    assert resp.steps == 3 and "step limit" in resp.answer
    assert registry.executed == ["find_best_model"]  # identical calls served from cache


def test_coverage_check_sends_model_back_for_missing_systems(registry):
    agent, llm = make_agent(
        registry,
        [
            calls(("find_best_model", {})),
            answer("premature answer"),
            calls(
                ("get_dataset_version", {}),
                ("get_dag_runs", {}),
                ("get_deployed_version", {}),
                ("get_retraining_policy", {}),
            ),
            answer("## Known facts\n- complete"),
        ],
    )
    resp = agent.run("Find the best model, its dataset, deployment status and whether retraining is necessary")
    assert "orchestrator check" in llm.seen[2][-1]["text"]
    assert resp.answer.startswith("## Known facts")
    assert required_modules("is retraining necessary?") == {"mlflow", "dvc", "kubernetes", "airflow", "policy"}


def test_api_query_confirm_and_validation(registry, monkeypatch):
    import api.main as api_main

    agent, _ = make_agent(
        registry, [calls(("trigger_training_pipeline", {})), answer("needs confirmation"), answer("- done")]
    )
    api_main.app.dependency_overrides[api_main.get_agent] = lambda: agent
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    client = TestClient(api_main.app)
    try:
        assert client.post("/agent/query", json={"message": "hi"}).status_code == 401
        h = {"X-API-Key": "test-key"}
        assert client.post("/agent/query", json={"message": ""}, headers=h).status_code == 422
        r = client.post("/agent/query", json={"message": "trigger training"}, headers=h)
        assert r.status_code == 200 and "X-Request-ID" in r.headers
        action_id = r.json()["pending_actions"][0]["action_id"]
        assert [p["action_id"] for p in client.get("/agent/pending", headers=h).json()] == [action_id]
        c = client.post("/agent/confirm", json={"action_id": action_id}, headers=h)
        assert c.json()["tool_calls"][0]["status"] == "ok"
    finally:
        api_main.app.dependency_overrides.clear()


def test_keyword_matching_uses_word_starts():
    assert required_modules("Is the rapid capital plan ok?") == set()  # 'api' inside words must not match
    assert required_modules("Why is my prediction API unavailable?") == {"kubernetes"}
    assert "kubernetes" in required_modules("check the deployments")  # prefix match still works


def test_action_requests_skip_the_coverage_nudge(registry):
    agent, llm = make_agent(registry, [calls(("trigger_training_pipeline", {})), answer("awaiting confirmation")])
    resp = agent.run("Trigger the model training pipeline")  # mentions 'model' (mlflow) but is an action request
    assert resp.steps == 2 and not any("orchestrator check" in str(m) for m in llm.seen[-1])


def test_log_redaction_keeps_token_counts_but_masks_secrets():
    from agent.logging_utils import redact

    out = redact(
        {
            "prompt_tokens": 1200,
            "output_tokens": 80,
            "access_token": "abc",
            "AIRFLOW_ADMIN_PASSWORD": "pw",
            "x-api-key": "k",
            "note": "password=hunter2 and AIzaSyA1234567890123456789012345678901",
        }
    )
    assert out["prompt_tokens"] == 1200 and out["output_tokens"] == 80
    assert {out["access_token"], out["AIRFLOW_ADMIN_PASSWORD"], out["x-api-key"]} == {"***REDACTED***"}
    assert "hunter2" not in out["note"] and "AIzaSy" not in out["note"]
