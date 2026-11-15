"""FastAPI interface for the MLOps agent.

    uvicorn api.main:app --port 8100

Endpoints
  GET  /health                  liveness of the API itself
  GET  /tools                   tool catalogue (name, module, schema, whether confirmation is required)
  POST /agent/query             natural-language request -> evidence-based answer + tool trace
  GET  /agent/pending           state-changing actions awaiting confirmation
  POST /agent/confirm           execute one pending action by id (explicit user confirmation)

If AGENT_API_KEY is set, every /agent and /tools call must send header `X-API-Key`.
"""

from __future__ import annotations

import hmac
import time
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.logging_utils import configure_logging, get_logger, log_event
from agent.orchestrator import MLOpsAgent, build_agent
from agent.schemas import AgentResponse, PendingAction, ToolSpec

configure_logging()
logger = get_logger("api")
app = FastAPI(
    title="MLOps AI Agent",
    version="1.0.0",
    description="Evidence-based operations agent over DVC, MLflow, Airflow and Kubernetes",
)


@lru_cache(maxsize=1)
def get_agent() -> MLOpsAgent:
    return build_agent()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().agent_api_key
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Natural-language request")
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Continue a conversation")
    include_evidence: bool = Field(False, description="Include raw tool results in the response")


class ConfirmRequest(BaseModel):
    action_id: str = Field(..., pattern=r"^[a-f0-9]{6,32}$")


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log_event(
        logger,
        "http_request",
        http_request_id=rid,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools", response_model=list[ToolSpec], dependencies=[Depends(require_api_key)])
def tools(agent: MLOpsAgent = Depends(get_agent)):
    return agent.registry.specs()


@app.post("/agent/query", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest, agent: MLOpsAgent = Depends(get_agent)):
    return agent.run(req.message, session_id=req.session_id, include_evidence=req.include_evidence)


@app.get("/agent/pending", response_model=list[PendingAction], dependencies=[Depends(require_api_key)])
def pending(agent: MLOpsAgent = Depends(get_agent)):
    return agent.list_pending()


@app.post("/agent/confirm", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def confirm(req: ConfirmRequest, agent: MLOpsAgent = Depends(get_agent)):
    return agent.confirm(req.action_id)
