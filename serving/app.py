"""Churn prediction API (FastAPI) - the ML application deployed to Kubernetes.

Endpoints
  GET  /health/live   liveness: the process is up and serving requests
  GET  /health/ready  readiness: the model is loaded and can predict (503 otherwise)
  GET  /version       app version + model lineage (registry version, run id, dataset version)
  POST /predict       churn probability and label for one or more customers

Configuration comes from environment variables populated by the ConfigMap (non-sensitive only).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

APP_VERSION = os.environ.get("APP_VERSION", "dev")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
STARTUP_DELAY_S = float(os.environ.get("STARTUP_DELAY_S", "0"))
THRESHOLD_OVERRIDE = os.environ.get("DECISION_THRESHOLD")

logging.basicConfig(
    stream=sys.stdout,
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("churn-api")

STATE: dict = {"model": None, "meta": {}, "error": None, "loaded_at": None}


def load_model() -> None:
    try:
        if STARTUP_DELAY_S:
            log.info("simulating slow model warm-up: %.0fs", STARTUP_DELAY_S)
            time.sleep(STARTUP_DELAY_S)
        meta_path = MODEL_DIR / "model_meta.json"
        model_path = MODEL_DIR / "mlflow_model"
        if not model_path.exists():
            raise FileNotFoundError(f"model directory not found: {model_path}")
        import mlflow.sklearn

        STATE["model"] = mlflow.sklearn.load_model(str(model_path))
        STATE["meta"] = json.loads(meta_path.read_text())
        STATE["loaded_at"] = time.time()
        STATE["error"] = None
        log.info(
            "model loaded: %s v%s (run %s, dataset %s)",
            STATE["meta"].get("model_name"),
            STATE["meta"].get("model_version"),
            STATE["meta"].get("mlflow_run_id"),
            STATE["meta"].get("dataset_version"),
        )
    except Exception as exc:  # noqa: BLE001 - keep the process alive so probes can report it
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log.error("MODEL_LOAD_FAILED %s", STATE["error"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("starting churn-api %s (model dir %s)", APP_VERSION, MODEL_DIR)
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", version=APP_VERSION, lifespan=lifespan)


class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0, le=600)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    num_products: int = Field(..., ge=1)
    internet_service: Literal["dsl", "fiber", "none"]
    senior_citizen: int = Field(..., ge=0, le=1)
    last_login_days: int | None = Field(None, ge=0)


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


def _features(customers: list[Customer], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump() for c in customers])
    df = pd.get_dummies(df, columns=["contract_type", "internet_service"], dtype=int)
    return df.reindex(columns=columns, fill_value=0).astype("float64")


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": STATE["error"] or "loading"})
    return {"status": "ready", "model_version": STATE["meta"].get("model_version")}


@app.get("/version")
def version():
    return {"app_version": APP_VERSION, "model": STATE["meta"], "model_loaded": STATE["model"] is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE['error'] or 'loading'}")
    columns = list(getattr(model, "feature_names_in_", []))
    X = _features(req.customers, columns)
    threshold = float(THRESHOLD_OVERRIDE) if THRESHOLD_OVERRIDE else STATE["meta"].get("decision_threshold", 0.5)
    proba = model.predict_proba(X)[:, 1]
    return {
        "model_version": STATE["meta"].get("model_version"),
        "threshold": threshold,
        "predictions": [{"churn_probability": round(float(p), 4), "churn": bool(p >= threshold)} for p in proba],
    }
