"""Structured JSON logging with secret redaction.

Every tool invocation emits one log record containing the tool name, status, duration and
request id. Values that look like credentials are masked before they reach any handler.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

SENSITIVE_KEYS = re.compile(r"(pass(word)?|secret|token|api[_-]?key|authorization|credential|kubeconfig)", re.I)
SENSITIVE_VALUES = [
    re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),  # LLM provider style keys
    re.compile(r"(?i)(://[^:/\s]+:)[^@/\s]+@"),  # user:password@ in URLs
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+"),  # JWTs
]
REDACTED = "***REDACTED***"


def redact(value: Any) -> Any:
    """Recursively mask secrets in dicts/lists/strings."""
    if isinstance(value, dict):
        return {k: (REDACTED if SENSITIVE_KEYS.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        out = value
        for pat in SENSITIVE_VALUES:
            out = pat.sub(lambda m: (m.group(1) + REDACTED + "@") if m.lastindex else REDACTED, out)
        return out
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str)


_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("mlops_agent")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = True  # lets pytest's caplog observe records
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mlops_agent.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": redact(fields)})
