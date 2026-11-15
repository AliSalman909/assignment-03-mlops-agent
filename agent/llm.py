"""LLM adapter layer. The orchestrator talks to `LLMClient`; Gemini is one implementation.

Messages are kept in a provider-neutral format so the agent loop is testable with a scripted LLM:
    {"role": "user",  "text": "..."}
    {"role": "model", "text": "...", "calls": [FunctionCall], "raw": <provider content>}
    {"role": "tool",  "results": [{"id", "name", "response": dict}]}
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent.config import Settings, get_settings
from agent.logging_utils import get_logger, log_event
from agent.schemas import ToolSpec

logger = get_logger("llm")


class LLMError(Exception):
    """The language model could not be reached or returned an unusable response."""


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class LLMTurn:
    text: str | None
    calls: list[FunctionCall] = field(default_factory=list)
    raw: Any = None


class LLMClient(Protocol):
    model_name: str

    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec] | None) -> LLMTurn: ...


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic JSON schema -> the subset Gemini function declarations accept reliably."""
    s = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node.pop("title", None)
            # Optional[X] -> anyOf[X, null]: keep X and mark nullable via description only
            if "anyOf" in node:
                options = [o for o in node["anyOf"] if o.get("type") != "null"]
                if len(options) == 1:
                    merged = {**options[0], **{k: v for k, v in node.items() if k != "anyOf"}}
                    node.clear()
                    node.update(merged)
            if node.get("default") is None:
                node.pop("default", None)
            for v in list(node.values()):
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        return node

    walk(s)
    s.setdefault("type", "object")
    s.setdefault("properties", {})
    return s


class GeminiClient:
    """Google Gemini via the google-genai SDK with manual (not automatic) function calling."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not configured")
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=int(self.settings.llm_timeout_s * 1000)),
        )
        self.model_name = self.settings.gemini_model
        self._tool_cache: dict[tuple[str, ...], Any] = {}

    def _tools(self, specs: list[ToolSpec]):
        key = tuple(s.name for s in specs)
        if key not in self._tool_cache:
            t = self._types
            decls = [
                t.FunctionDeclaration(
                    name=s.name,
                    description=s.description + (" [REQUIRES USER CONFIRMATION]" if s.requires_confirmation else ""),
                    parameters_json_schema=_clean_schema(s.input_schema),
                )
                for s in specs
            ]
            self._tool_cache[key] = [t.Tool(function_declarations=decls)]
        return self._tool_cache[key]

    def _contents(self, messages: list[dict[str, Any]]):
        t = self._types
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part.from_text(text=m["text"])]))
            elif m["role"] == "model":
                if m.get("raw") is not None:
                    out.append(m["raw"])  # preserves thought signatures for multi-turn function calling
                else:
                    parts = [t.Part.from_text(text=m["text"])] if m.get("text") else []
                    parts += [t.Part.from_function_call(name=c.name, args=c.args) for c in m.get("calls", [])]
                    out.append(t.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                out.append(
                    t.Content(
                        role="user",
                        parts=[
                            t.Part.from_function_response(name=r["name"], response=r["response"]) for r in m["results"]
                        ],
                    )
                )
        return out

    def generate(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec] | None) -> LLMTurn:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            tools=self._tools(tools) if tools else None,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        start = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=self.model_name, contents=self._contents(messages), config=config
            )
        except Exception as exc:  # noqa: BLE001 - surface as a typed, non-secret error
            log_event(
                logger,
                "llm_call",
                status="error",
                model=self.model_name,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                error=type(exc).__name__,
            )
            raise LLMError(f"Gemini request failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        if not resp.candidates:
            raise LLMError("Gemini returned no candidates (possibly blocked)")
        content = resp.candidates[0].content
        calls = [
            FunctionCall(name=fc.name, args=dict(fc.args or {}), id=getattr(fc, "id", None))
            for fc in (resp.function_calls or [])
        ]
        text = None
        if content and content.parts:
            text = "".join(
                p.text for p in content.parts if getattr(p, "text", None) and not getattr(p, "thought", False)
            )
        usage = getattr(resp, "usage_metadata", None)
        log_event(
            logger,
            "llm_call",
            status="ok",
            model=getattr(resp, "model_version", self.model_name),
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            function_calls=len(calls),
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
        return LLMTurn(text=text or None, calls=calls, raw=content)
