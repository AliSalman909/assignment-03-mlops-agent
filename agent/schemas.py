"""Shared structured schemas for tool inputs/outputs.

Every tool returns a `ToolResult` envelope so the orchestrator (and the LLM) can always
distinguish real evidence (`status == "ok"`) from failures, and never has to guess.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"  # tool ran but the operation failed (e.g. git rev not found)
    UNAVAILABLE = "unavailable"  # backing service/binary not reachable
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"  # arguments failed validation / not allowed
    CONFIRMATION_REQUIRED = "confirmation_required"  # mutating op without explicit user consent
    FORBIDDEN = "forbidden"  # permission policy denies the operation outright


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    request_id: str
    duration_ms: float
    data: dict[str, Any] | None = None
    error: str | None = None
    # human-readable provenance of where the evidence came from (commands, APIs)
    evidence_source: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.OK


class ToolSpec(BaseModel):
    """Provider-agnostic tool description handed to the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]
    mutating: bool = False
    requires_confirmation: bool = False
    module: str
