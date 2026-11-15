"""Populate MLflow with a realistic, provenance-tagged experiment history.

For each dataset version tag (data-v1.0, data-v1.1, data-v2.0) the exact bytes are read through
the DVC API at that git commit, then three model families are trained and logged to the
`churn-baseline` experiment with git_commit / dataset_md5 / dataset_version tags.

The best v1.1 model is registered as `churn-classifier` version 1 with alias `champion`
(= what is currently deployed). Newer data (v2.0) exists that the champion never saw, which
gives the agent something real to reason about.

    python scripts/seed_experiments.py            # requires MLflow at MLFLOW_TRACKING_URI
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import dvc.api
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.preprocess import preprocess  # noqa: E402
from training.train import (  # noqa: E402
    REGISTERED_MODEL_NAME,
    log_to_mlflow,
    split,
    train_and_evaluate,
)

DATASET = "data/raw/churn.csv"
EXPERIMENT = "churn-baseline"
VERSIONS = ["v1.0", "v1.1", "v2.0"]
MODELS = [
    ("logistic_regression", {"C": 0.5}),
    ("random_forest", {"n_estimators": 300, "max_depth": 10}),
    ("gradient_boosting", {"n_estimators": 200, "learning_rate": 0.05}),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT) and client.search_runs(
        [client.get_experiment_by_name(EXPERIMENT).experiment_id], max_results=1
    ):
        print(f"experiment '{EXPERIMENT}' already has runs - nothing to do")
        return

    results = []
    for version in VERSIONS:
        tag = f"data-{version}"
        commit = git("rev-parse", f"{tag}^{{commit}}")
        md5 = yaml.safe_load(git("show", f"{commit}:{DATASET}.dvc"))["outs"][0]["md5"]
        raw = pd.read_csv(io.StringIO(dvc.api.read(DATASET, repo=str(ROOT), rev=commit, encoding="utf-8")))
        train, test = split(preprocess(raw))
        for model_type, params in MODELS:
            model, used, metrics = train_and_evaluate(train, test, model_type, params)
            tags = {
                "git_commit": commit,
                "dataset_md5": md5,
                "dataset_path": DATASET,
                "dataset_version": version,
                "source": "seed_experiments",
                "quality_gate": "not_evaluated",
            }
            out = log_to_mlflow(
                model,
                used,
                metrics,
                experiment=EXPERIMENT,
                tags=tags,
                train=train,
                run_name=f"{model_type}-data-{version}",
            )
            results.append((version, model_type, metrics, out))
            print(f"{version} {model_type:20} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f} run={out['run_id']}")

    # champion = best F1 among models trained on v1.1 (what is "in production")
    v11 = [r for r in results if r[0] == "v1.1"]
    version, model_type, metrics, out = max(v11, key=lambda r: r[2]["f1"])
    mv = mlflow.register_model(out["model_uri"], REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
    client.set_tag(out["run_id"], "quality_gate", "passed")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "dataset_version", version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "source_run_id", out["run_id"])
    client.update_model_version(
        REGISTERED_MODEL_NAME,
        mv.version,
        description=f"{model_type} trained on dataset {version}; promoted after review.",
    )
    print(f"registered {REGISTERED_MODEL_NAME} v{mv.version} ({model_type}, data {version}) as 'champion'")


if __name__ == "__main__":
    main()
