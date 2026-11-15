"""Tool registry + safe execution wrapper shared by all four modules.

`execute_tool` is the single choke point every tool call passes through. It:
  * validates arguments against the tool's pydantic input model,
  * enforces the confirmation policy for state-changing tools,
  * converts every failure mode into a typed `ToolResult` (never raises to the LLM),
  * emits one structured log line (tool, status, duration, request id - no secrets).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.logging_utils import get_logger, log_event
from agent.schemas import ToolResult, ToolSpec, ToolStatus

logger = get_logger("tools")


# ---------------------------------------------------------------------------
# Exceptions tools raise; execute_tool maps them to ToolStatus values.
# ---------------------------------------------------------------------------
class ToolError(Exception):
    """Operation ran but failed (bad revision, missing file, ...)."""


class ToolUnavailableError(ToolError):
    """Backing service or binary is not reachable."""


class ToolTimeoutError(ToolError):
    """Operation exceeded its time budget."""


class ToolForbiddenError(ToolError):
    """Operation is outside the permission policy (e.g. command not allow-listed)."""


class ToolInvalidRequestError(ToolError):
    """Arguments are syntactically valid but semantically unacceptable."""


@dataclass
class ToolOutput:
    data: dict[str, Any]
    evidence_source: list[str]


@dataclass
class Tool:
    name: str
    module: str
    description: str
    input_model: type[BaseModel]
    func: Callable[[BaseModel], ToolOutput]
    mutating: bool = False
    requires_confirmation: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            module=self.module,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            mutating=self.mutating,
            requires_confirmation=self.requires_confirmation,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self, module: str | None = None) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values() if module is None or t.module == module]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        request_id: str | None = None,
        confirmed: bool = False,
    ) -> ToolResult:
        return execute_tool(self, name, arguments or {}, request_id=request_id, confirmed=confirmed)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def execute_tool(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
    *,
    request_id: str | None = None,
    confirmed: bool = False,
) -> ToolResult:
    request_id = request_id or new_request_id()
    start = time.perf_counter()
    tool = registry.get(name)

    def finish(status: ToolStatus, data=None, error=None, sources=None) -> ToolResult:
        duration = round((time.perf_counter() - start) * 1000, 2)
        level = logging.INFO if status == ToolStatus.OK else logging.WARNING
        log_event(
            logger,
            "tool_call",
            level=level,
            tool=name,
            module=tool.module if tool else None,
            status=status.value,
            duration_ms=duration,
            request_id=request_id,
            error=error,
        )
        return ToolResult(
            tool=name,
            status=status,
            request_id=request_id,
            duration_ms=duration,
            data=data,
            error=error,
            evidence_source=sources or [],
        )

    if tool is None:
        return finish(ToolStatus.INVALID_REQUEST, error=f"unknown tool '{name}'")

    try:
        params = tool.input_model.model_validate(arguments)
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors())
        return finish(ToolStatus.INVALID_REQUEST, error=f"invalid arguments: {msgs}")

    if tool.requires_confirmation and not confirmed:
        return finish(
            ToolStatus.CONFIRMATION_REQUIRED,
            error=f"'{name}' changes state and requires explicit user confirmation; nothing was executed",
        )

    try:
        out = tool.func(params)
    except ToolTimeoutError as exc:
        return finish(ToolStatus.TIMEOUT, error=str(exc))
    except ToolUnavailableError as exc:
        return finish(ToolStatus.UNAVAILABLE, error=str(exc))
    except ToolForbiddenError as exc:
        return finish(ToolStatus.FORBIDDEN, error=str(exc))
    except ToolInvalidRequestError as exc:
        return finish(ToolStatus.INVALID_REQUEST, error=str(exc))
    except ToolError as exc:
        return finish(ToolStatus.ERROR, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - last line of defence; never crash the agent loop
        logger.exception("unexpected tool failure", extra={"fields": {"tool": name, "request_id": request_id}})
        return finish(ToolStatus.ERROR, error=f"unexpected {type(exc).__name__}: {exc}")

    return finish(ToolStatus.OK, data=out.data, sources=out.evidence_source)
