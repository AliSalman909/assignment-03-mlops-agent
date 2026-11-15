"""Shared training library used by the Airflow DAG, the experiment seeding script and the CLI.

Provenance contract (read by the agent's MLflow + DVC tools):
    every MLflow run is tagged with
        git_commit     - commit whose .dvc pointer describes the training data
        dataset_md5    - md5 of the exact raw bytes trained on (== DVC's recorded md5)
        dataset_path   - repo-relative path of the dataset
        dataset_version- label if known (e.g. v2.0), otherwise resolved later by the DVC tools
    so "which dataset trained run X?" is answered from recorded evidence, not guessed.

CLI:
    python training/train.py --data data/raw/churn.csv --model-type gradient_boosting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.preprocess import TARGET, preprocess

log = logging.getLogger("training")

REQUIRED_COLUMNS = [
    "customer_id",
    "tenure_months",
    "contract_type",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "num_products",
    "internet_service",
    "senior_citizen",
    TARGET,
]

# Defaults for the quality gate; the DAG lets a run override them via conf.
DEFAULT_GATE = {"min_f1": 0.48, "min_roc_auc": 0.70, "max_f1_regression": 0.02}  # calibrated on seeded runs
REGISTERED_MODEL_NAME = "churn-classifier"
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


# ----------------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1000) -> ValidationReport:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    check("required_columns", not missing, f"missing={missing}" if missing else "all present")
    check("min_rows", len(df) >= min_rows, f"rows={len(df)} (min {min_rows})")
    null_rate = float(df.isna().mean().max()) if len(df) else 1.0
    check("null_rate", null_rate <= 0.05, f"max column null rate={null_rate:.3f} (max 0.05)")
    dup_rate = float(df.duplicated().mean()) if len(df) else 0.0
    check("duplicate_rate", dup_rate <= 0.05, f"duplicate rows={dup_rate:.3%} (max 5%)")
    pos_rate = None
    if TARGET in df.columns:
        labels = set(pd.unique(df[TARGET].dropna()))
        check("binary_label", labels <= {0, 1}, f"label values={sorted(map(str, labels))}")
        pos_rate = float(df[TARGET].mean())
        check("class_balance", 0.05 <= pos_rate <= 0.60, f"positive rate={pos_rate:.3f} (0.05-0.60)")
    if "monthly_charges" in df.columns:
        neg = int((df["monthly_charges"] < 0).sum())
        check("non_negative_charges", neg == 0, f"negative monthly_charges rows={neg}")

    stats = {"rows": int(len(df)), "columns": int(df.shape[1]), "positive_rate": pos_rate, "duplicate_rate": dup_rate}
    return ValidationReport(passed=all(c["passed"] for c in checks), checks=checks, stats=stats)


# ----------------------------------------------------------------------------- provenance
def file_md5(path: str | Path) -> str:
    h = hashlib.md5()  # noqa: S324 - matches DVC content addressing
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_git_commit(repo_root: str | Path) -> str | None:
    """Resolve HEAD without needing the git binary (works inside the Airflow container)."""
    git = Path(repo_root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text().strip()
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


# ----------------------------------------------------------------------------- modelling
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1},
    "gradient_boosting": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
}


def build_model(model_type: str, params: dict[str, Any] | None = None):
    if model_type not in MODEL_DEFAULTS:
        raise ValueError(f"unknown model_type '{model_type}'; choose from {sorted(MODEL_DEFAULTS)}")
    p = {**MODEL_DEFAULTS[model_type], **(params or {})}
    if model_type == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**p))]), p
    if model_type == "random_forest":
        return RandomForestClassifier(class_weight="balanced", **p), p
    return GradientBoostingClassifier(**p), p


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])


def evaluate(model, test: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    X, y = test.drop(columns=[TARGET]), test[TARGET]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "f1": round(float(f1_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def best_threshold(model, valid: pd.DataFrame) -> float:
    """Pick the probability threshold maximising F1 on a validation split (class imbalance)."""
    X, y = valid.drop(columns=[TARGET]), valid[TARGET]
    proba = model.predict_proba(X)[:, 1]
    grid = np.arange(0.2, 0.71, 0.05)
    scores = [f1_score(y, (proba >= t).astype(int)) for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def train_and_evaluate(
    train: pd.DataFrame, test: pd.DataFrame, model_type: str, params: dict[str, Any] | None = None
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    fit_part, valid_part = split(train, test_size=0.2, seed=7)
    model, used = build_model(model_type, params)
    model.fit(fit_part.drop(columns=[TARGET]), fit_part[TARGET])
    threshold = best_threshold(model, valid_part)
    model.fit(train.drop(columns=[TARGET]), train[TARGET])  # refit on all training data
    metrics = evaluate(model, test, threshold)
    return model, {**used, "model_type": model_type, "decision_threshold": threshold}, metrics


# ----------------------------------------------------------------------------- MLflow
def log_to_mlflow(
    model,
    params: dict[str, Any],
    metrics: dict[str, float],
    *,
    experiment: str,
    tags: dict[str, str],
    train: pd.DataFrame,
    run_name: str | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> dict[str, str]:
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in params.items() if k != "n_jobs"})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_metric("train_rows", len(train))
        for name, payload in (extra_artifacts or {}).items():
            mlflow.log_dict(payload, name)
        signature_input = train.drop(columns=[TARGET]).head(5).astype("float64")
        # MLflow 3.18 serialises with skops, which only loads explicitly trusted types. We trained these
        # tree models ourselves in this process, so trusting sklearn's Tree node storage is safe.
        info = mlflow.sklearn.log_model(
            model, name="model", input_example=signature_input, skops_trusted_types=SKOPS_TRUSTED_TYPES
        )
        return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id, "model_uri": info.model_uri}


def gate_decision(metrics: dict[str, float], champion_f1: float | None, gate: dict[str, float]) -> dict[str, Any]:
    reasons = []
    if metrics["f1"] < gate["min_f1"]:
        reasons.append(f"f1 {metrics['f1']:.4f} < min_f1 {gate['min_f1']}")
    if metrics["roc_auc"] < gate["min_roc_auc"]:
        reasons.append(f"roc_auc {metrics['roc_auc']:.4f} < min_roc_auc {gate['min_roc_auc']}")
    if champion_f1 is not None and metrics["f1"] < champion_f1 - gate["max_f1_regression"]:
        reasons.append(
            f"f1 {metrics['f1']:.4f} regresses more than {gate['max_f1_regression']} vs champion {champion_f1:.4f}"
        )
    return {"passed": not reasons, "reasons": reasons, "thresholds": gate, "champion_f1": champion_f1}


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Train a churn model and log it to MLflow")
    ap.add_argument("--data", default="data/raw/churn.csv")
    ap.add_argument("--model-type", default="gradient_boosting", choices=sorted(MODEL_DEFAULTS))
    ap.add_argument("--params", default="{}", help="JSON hyper-parameter overrides")
    ap.add_argument("--experiment", default="churn-manual")
    ap.add_argument("--dataset-version", default="")
    args = ap.parse_args()

    raw = pd.read_csv(args.data)
    report = validate_dataframe(raw)
    if not report.passed:
        raise SystemExit(f"validation failed: {report.failures}")
    train, test = split(preprocess(raw))
    model, params, metrics = train_and_evaluate(train, test, args.model_type, json.loads(args.params))
    tags = {
        "git_commit": read_git_commit(Path(__file__).resolve().parent.parent) or "unknown",
        "dataset_md5": file_md5(args.data),
        "dataset_path": args.data.replace("\\", "/"),
        "dataset_version": args.dataset_version or "unresolved",
        "source": "training/train.py",
    }
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    out = log_to_mlflow(model, params, metrics, experiment=args.experiment, tags=tags, train=train)
    print(json.dumps({**out, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
