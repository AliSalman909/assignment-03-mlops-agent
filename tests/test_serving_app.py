"""The inference service must report readiness honestly (probes drive Kubernetes routing)."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "missing"))
    import serving.app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as client:
        yield client


def test_liveness_ok_but_readiness_fails_without_model(client_without_model):
    assert client_without_model.get("/health/live").status_code == 200
    ready = client_without_model.get("/health/ready")
    assert ready.status_code == 503
    assert "model directory not found" in ready.json()["reason"]


def test_predict_refuses_instead_of_guessing_without_model(client_without_model):
    body = {
        "customers": [
            {
                "tenure_months": 3,
                "contract_type": "month-to-month",
                "monthly_charges": 90,
                "total_charges": 270,
                "support_tickets": 2,
                "num_products": 1,
                "internet_service": "fiber",
                "senior_citizen": 0,
            }
        ]
    }
    r = client_without_model.post("/predict", json=body)
    assert r.status_code == 503
    assert client_without_model.get("/version").json()["model_loaded"] is False


def test_predict_validates_input(client_without_model):
    r = client_without_model.post("/predict", json={"customers": [{"tenure_months": -1}]})
    assert r.status_code == 422
