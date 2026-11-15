"""MLOps agent orchestrator: LLM tool-calling loop over the DVC / MLflow / Airflow / Kubernetes tools.

Guarantees enforced in code (not just in the prompt):
  * The LLM can only *request* state-changing tools. Those requests are parked as pending
    actions; they run only when the user confirms that exact action id (`confirm`). A
    confirmation is single-use and expires.
  * Every tool call goes through `ToolRegistry.execute` (validation, permissions, typed status,
    structured logs with the request id).
  * Failed / unavailable tools are listed in `data_gaps` and appended to the answer, so a
    response can never silently present missing evidence as fact.
  * Loop and tool-call budgets bound cost and runaway behaviour; identical calls are cached.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.config import Settings, get_settings
from agent.llm import FunctionCall, LLMClient, LLMError
from agent.logging_utils import get_logger, log_event
from agent.schemas import AgentResponse, PendingAction, ToolCallRecord, ToolResult, ToolStatus
from agent.tools.base import ToolRegistry, new_request_id

logger = get_logger("orchestrator")

SYSTEM_PROMPT = """You are an MLOps operations agent for a customer-churn ML system. You answer questions by calling
tools that query DVC (dataset versions), MLflow (experiments, runs, model registry), Airflow (training pipeline)
and Kubernetes (the deployed prediction API), plus the retraining policy.

EVIDENCE RULES (mandatory)
- Every operational claim (versions, metrics, statuses, counts, ids, health) must come from a tool result in this
  conversation. Never use prior knowledge or assumptions for operational facts.
- If a tool returns status other than "ok" (unavailable, timeout, error, forbidden, invalid_request), that fact is
  UNKNOWN. Say so explicitly; do not guess or fill in plausible values.
- Quote concrete identifiers (run ids, versions, commits, pod names) from tool results.

TOOL USE
- Plan which tools are needed, then call them. Call independent tools in parallel when possible.
- Dataset provenance: MLflow runs and registry versions carry provenance tags git_commit and dataset_md5. To learn
  which dataset version trained a model, call get_dataset_version(git_commit=<commit>, expected_md5=<md5>).
  Never infer the dataset version from a run name or a tag that says "unresolved".
- The deployed model is the registry alias 'champion'; 'candidate' versions passed the automated gate but are not
  deployed. Use get_deployed_version and diagnose_service for deployment health.
- The highest offline metric does NOT make a model production-ready; mention margin, trade-offs, quality gate and
  registry status.

SAFETY
- Tools marked [REQUIRES USER CONFIRMATION] change state (trigger training, restart/scale/rollback, restore data).
  Only call them when the user explicitly asks for that action. Calling one only creates a pending action that
  the user must confirm; never claim it has run. Tell the user the action id is awaiting confirmation.
- Never trigger retraining because you think it would help. Questions like "is retraining necessary?" are answered
  with a recommendation only.
- When the user asks to trigger the pipeline and report the outcome, call trigger_training_pipeline with
  wait_seconds=600.

RETRAINING / "IS THE MODEL OK" QUESTIONS - gather ALL of this before answering:
  1. MLflow: find_best_model (best recent run) and get_model_versions (champion = deployed, candidate).
  2. DVC: get_dataset_version(git_commit, expected_md5) for the best run AND for the champion's source run, and
     get_dataset_history to learn the latest dataset version. (MLflow's dataset_version_tag_unverified is NOT
     evidence.)
  3. Kubernetes: get_deployed_version and diagnose_service (health, which model version is actually served).
  4. Airflow: get_dag_runs for churn_training_pipeline, and diagnose_dag_run for the most recent failed run if any.
  5. get_retraining_policy, then evaluate EACH rule explicitly as triggered / not triggered / unknown, citing evidence.
- Recommend retraining only when a rule is triggered by evidence. If a 'candidate' trained on the latest data
  already passed the quality gate, recommend reviewing/promoting it instead of retraining.

ANSWER FORMAT for investigations and multi-step questions (markdown):
## Known facts
(bullet points, each with its source tool)
## Uncertainties
(what is unknown or not measured, and why)
## Recommended actions
(numbered; state that nothing was executed unless the user confirmed an action. Each action must address the
diagnosed root cause - prefer the `action`/`recommended_action` fields returned by diagnostic tools. Do not suggest
unrelated remediation, e.g. a rollback when the cause is configuration that a rollback would not change.)
Keep it concise. For simple lookups a short direct answer with sources is fine."""


# Which tool modules a question needs, inferred from its wording. Used for a single coverage check
# before the final answer is accepted, so multi-system questions are not answered from partial evidence.
MODULE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mlflow": ("model", "experiment", "run", "metric", "f1", "registry", "champion", "candidate", "retrain"),
    "dvc": ("dataset", "data version", "dvc", "provenance", "trained on", "retrain"),
    "kubernetes": ("deploy", "kubernetes", "k8s", "pod", "service", "api", "serving", "replica", "retrain"),
    "airflow": ("pipeline", "airflow", "dag", "workflow", "training run", "retrain"),
    "policy": ("retrain",),
}


_KEYWORD_RES = {m: re.compile(r"\b(" + "|".join(map(re.escape, ws)) + r")") for m, ws in MODULE_KEYWORDS.items()}


def required_modules(message: str) -> set[str]:
    """Modules a question involves. Keywords match at word starts ('deploy' matches 'deployment', but 'api'
    does not match 'rapid')."""
    text = message.lower()
    return {m for m, rx in _KEYWORD_RES.items() if rx.search(text)}


@dataclass
class _Parked:
    action: PendingAction
    request_id: str
    session_id: str


@dataclass
class _Session:
    messages: list[dict[str, Any]] = field(default_factory=list)
    updated: float = field(default_factory=time.time)


class MLOpsAgent:
    def __init__(self, registry: ToolRegistry, llm: LLMClient, settings: Settings | None = None) -> None:
        self.registry = registry
        self.llm = llm
        self.settings = settings or get_settings()
        self.specs = registry.specs()
        self._pending: dict[str, _Parked] = {}
        self._sessions: dict[str, _Session] = {}
        # FastAPI runs sync endpoints in a thread pool: guard shared state
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ helpers
    def _payload(self, result: ToolResult) -> dict[str, Any]:
        """What the LLM sees for a tool result: status first, data truncated to a budget."""
        body = {"status": result.status.value, "error": result.error, "evidence_source": result.evidence_source}
        if result.data is not None:
            data = json.dumps(result.data, default=str)
            limit = self.settings.agent_tool_result_chars
            if len(data) > limit:
                body["data_truncated"] = data[:limit] + "...[truncated]"
            else:
                body["data"] = result.data
        else:
            body["data"] = None
            body["note"] = "no data returned - treat the requested facts as UNKNOWN"
        return body

    def _park(self, call: FunctionCall, request_id: str, session_id: str) -> PendingAction:
        with self._lock:
            now = time.time()
            for aid in [a for a, p in self._pending.items() if p.action.expires_at < now]:
                del self._pending[aid]
        spec = next(s for s in self.specs if s.name == call.name)
        action = PendingAction(
            action_id=uuid.uuid4().hex[:10],
            tool=call.name,
            arguments=call.args,
            description=spec.description.split(".")[0],
            expires_at=time.time() + self.settings.pending_action_ttl_s,
        )
        with self._lock:
            self._pending[action.action_id] = _Parked(action, request_id, session_id)
        return action

    def _model_label(self) -> str:
        name = getattr(self.llm, "model_name", "unknown")
        resolved = getattr(self.llm, "last_model_version", None)
        return f"{name} ({resolved})" if resolved and resolved != name else name

    @staticmethod
    def _gap(record: ToolCallRecord) -> str:
        return f"{record.tool}: {record.status.value} - {record.error or 'no data'}"

    MAX_SESSIONS = 200
    MAX_HISTORY = 40

    def _session(self, session_id: str | None) -> tuple[str, _Session]:
        with self._lock:
            now = time.time()
            for sid in [s for s, v in self._sessions.items() if now - v.updated > 3600]:
                del self._sessions[sid]  # expire idle sessions
            while len(self._sessions) >= self.MAX_SESSIONS and session_id not in self._sessions:
                oldest = min(self._sessions, key=lambda k: self._sessions[k].updated)
                del self._sessions[oldest]  # bounded memory: evict least recently used
            sid = session_id or uuid.uuid4().hex[:12]
            return sid, self._sessions.setdefault(sid, _Session())

    @classmethod
    def _trim(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep recent history, cutting only at a user turn so a function response is never separated from
        the function call it answers (Gemini rejects orphaned function responses)."""
        if len(messages) <= cls.MAX_HISTORY:
            return messages
        start = len(messages) - cls.MAX_HISTORY
        while start < len(messages) and messages[start]["role"] != "user":
            start += 1
        return messages[start:]

    # ------------------------------------------------------------------ main loop
    def run(self, message: str, session_id: str | None = None, include_evidence: bool = False) -> AgentResponse:
        request_id = new_request_id()
        sid, session = self._session(session_id)
        messages = session.messages + [{"role": "user", "text": message}]
        records: list[ToolCallRecord] = []
        evidence: list[ToolResult] = []
        pending: list[PendingAction] = []
        cache: dict[str, ToolResult] = {}
        answer: str | None = None
        nudged = False
        skipped: list[str] = []
        step = 0
        log_event(logger, "agent_request", request_id=request_id, session_id=sid, message_chars=len(message))

        while step < self.settings.agent_max_steps:
            step += 1
            try:
                turn = self.llm.generate(SYSTEM_PROMPT, messages, self.specs)
            except LLMError as exc:
                log_event(logger, "agent_llm_failure", request_id=request_id, error=str(exc)[:200])
                answer = (
                    "I could not complete this request because the language model is unavailable "
                    f"({exc}). No conclusions were drawn."
                )
                if evidence:
                    answer += " Tool results gathered before the failure are listed in the evidence."
                break
            messages.append({"role": "model", "text": turn.text, "calls": turn.calls, "raw": turn.raw})
            if not turn.calls:
                used = {s.module for s in self.specs if s.name in {r.tool for r in records}}
                missing = sorted(required_modules(message) - used)
                if missing and not nudged and not pending and step < self.settings.agent_max_steps:
                    nudged = True
                    log_event(logger, "agent_coverage_nudge", request_id=request_id, missing=missing)
                    messages.append(
                        {
                            "role": "user",
                            "text": (
                                "[orchestrator check] Before answering, gather evidence from the systems this question "
                                f"involves that you have not queried yet: {', '.join(missing)}. "
                                "Call the relevant tools, "
                                "then give the complete answer in the required format."
                            ),
                        }
                    )
                    continue
                answer = turn.text or "(the model returned an empty answer)"
                break

            results = []
            for call in turn.calls:
                key = f"{call.name}:{json.dumps(call.args, sort_keys=True, default=str)}"
                if len(records) >= self.settings.agent_max_tool_calls:
                    payload = {"status": "error", "error": "tool-call budget exhausted; answer with what you have"}
                    skipped.append(f"{call.name}: not called - tool-call budget exhausted")
                elif key in cache:
                    payload = self._payload(cache[key]) | {"note": "cached result of an identical earlier call"}
                else:
                    res = self.registry.execute(call.name, call.args, request_id=request_id, confirmed=False)
                    if res.status == ToolStatus.CONFIRMATION_REQUIRED:
                        action = self._park(call, request_id, sid)
                        pending.append(action)
                        payload = {
                            "status": "pending_user_confirmation",
                            "action_id": action.action_id,
                            "note": "NOT executed. Tell the user this action is awaiting their explicit confirmation.",
                        }
                    else:
                        cache[key] = res
                        evidence.append(res)
                        payload = self._payload(res)
                    records.append(
                        ToolCallRecord(
                            step=step,
                            tool=call.name,
                            arguments=call.args,
                            status=res.status,
                            duration_ms=res.duration_ms,
                            error=res.error,
                        )
                    )
                results.append({"id": call.id, "name": call.name, "response": payload})
            messages.append({"role": "tool", "results": results})
        else:
            answer = (
                "I stopped after reaching the step limit before finishing. The evidence gathered so far is "
                "listed below; please narrow the question."
            )

        gaps = [
            self._gap(r) for r in records if r.status not in (ToolStatus.OK, ToolStatus.CONFIRMATION_REQUIRED)
        ] + skipped
        if gaps:
            answer = f"{answer}\n\n**Data gaps (tools that did not return evidence):**\n" + "\n".join(
                f"- {g}" for g in gaps
            )
        if pending:
            answer += "\n\n**Awaiting your confirmation (not executed):**\n" + "\n".join(
                f"- `{a.action_id}`: {a.tool}({json.dumps(a.arguments)})" for a in pending
            )

        # keep a bounded history so follow-up questions have context
        session.messages = self._trim(messages)
        session.updated = time.time()
        log_event(
            logger,
            "agent_response",
            request_id=request_id,
            session_id=sid,
            steps=step,
            tool_calls=len(records),
            pending=len(pending),
            data_gaps=len(gaps),
        )
        return AgentResponse(
            request_id=request_id,
            session_id=sid,
            answer=answer,
            model=self._model_label(),
            steps=step,
            tool_calls=records,
            pending_actions=pending,
            data_gaps=gaps,
            evidence=evidence if include_evidence else None,
        )

    # ------------------------------------------------------------------ confirmation
    def list_pending(self) -> list[PendingAction]:
        now = time.time()
        with self._lock:
            return [p.action for p in self._pending.values() if p.action.expires_at > now]

    def confirm(self, action_id: str, summarize: bool = True) -> AgentResponse:
        """Execute exactly the parked action (same tool, same arguments) - once."""
        with self._lock:
            parked = self._pending.pop(action_id, None)  # atomic: an action can only be confirmed once
        request_id = new_request_id()
        if parked is None or parked.action.expires_at < time.time():
            return AgentResponse(
                request_id=request_id,
                session_id=parked.session_id if parked else "-",
                answer=f"No pending action '{action_id}' (unknown, already used, or expired). Nothing was executed.",
                model=self._model_label(),
                steps=0,
            )
        a = parked.action
        log_event(
            logger,
            "action_confirmed",
            request_id=request_id,
            action_id=action_id,
            tool=a.tool,
            origin_request_id=parked.request_id,
        )
        res = self.registry.execute(a.tool, a.arguments, request_id=request_id, confirmed=True)
        record = ToolCallRecord(
            step=1, tool=a.tool, arguments=a.arguments, status=res.status, duration_ms=res.duration_ms, error=res.error
        )
        answer = f"Executed `{a.tool}` after your confirmation: status **{res.status.value}**."
        if res.error:
            answer += f" Error: {res.error}"
        if summarize:
            try:
                turn = self.llm.generate(
                    "Summarise this tool result for an ML engineer in 3-6 bullet points. Use only facts from the "
                    "result; if the status is not 'ok', say clearly that it did not succeed.",
                    [{"role": "user", "text": json.dumps(self._payload(res), default=str)}],
                    None,
                )
                if turn.text:
                    answer += "\n\n" + turn.text
            except LLMError:
                pass
        _, session = self._session(parked.session_id)
        session.messages.append(
            {
                "role": "user",
                "text": f"[system note] The user confirmed action {action_id} "
                f"({a.tool}); result status: {res.status.value}; data: "
                f"{json.dumps(self._payload(res), default=str)[:3000]}",
            }
        )
        session.messages.append({"role": "model", "text": "Noted.", "calls": [], "raw": None})
        return AgentResponse(
            request_id=request_id,
            session_id=parked.session_id,
            answer=answer,
            model=self._model_label(),
            steps=1,
            tool_calls=[record],
            data_gaps=[] if res.ok else [self._gap(record)],
            evidence=[res],
        )


def build_agent(settings: Settings | None = None) -> MLOpsAgent:
    from agent.llm import GeminiClient
    from agent.tools import build_registry

    s = settings or get_settings()
    return MLOpsAgent(build_registry(), GeminiClient(s), s)
