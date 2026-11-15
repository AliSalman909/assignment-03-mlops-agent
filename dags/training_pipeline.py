"""Churn model training pipeline (Airflow 3).

validate_dataset -> preprocess_data -> train_model -> evaluate_and_log -> quality_gate -> register_candidate

* Retries: every task retries twice (10s back-off) for transient errors. Deterministic failures
  (bad data, failed gate) raise AirflowFailException, which fails immediately without retrying.
* Promotion safety: register_candidate only runs if *every* upstream task succeeded. A failed
  validation or quality gate leaves it `upstream_failed`, so no model version is registered.
  Even on success the model only gets the `candidate` alias; promotion to `champion`
  (what serving uses) is a human decision.
* Provenance: the MLflow run is tagged with git_commit + dataset_md5 (+ Airflow ids).
* The project is mounted read-only at /opt/project; intermediate files go to /opt/airflow/work.

Trigger-time conf (all optional):
    dataset_path     repo-relative dataset (default data/raw/churn.csv)
    model_type       logistic_regression | random_forest | gradient_boosting
    min_f1, min_roc_auc   quality-gate thresholds
    inject_failure   none | corrupt_data | transient_error   (for failure-diagnosis demos)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # pragma: no cover - older 3.x layout
    from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("MLOPS_PROJECT_ROOT", "/opt/project"))
WORK_ROOT = Path(os.environ.get("MLOPS_WORK_ROOT", "/opt/airflow/work"))
EXPERIMENT = "churn-training-pipeline"


def _workdir() -> Path:
    ctx = get_current_context()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ctx["dag_run"].run_id)
    d = WORK_ROOT / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _conf() -> dict:
    return dict(get_current_context()["params"])


@dag(
    dag_id="churn_training_pipeline",
    description="Validate -> preprocess -> train -> evaluate/MLflow -> quality gate -> register candidate",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training", "approved"],
    default_args={
        "owner": "mlops",
        "retries": 2,
        "retry_delay": timedelta(seconds=10),
        "execution_timeout": timedelta(minutes=15),
    },
    params={
        "dataset_path": Param("data/raw/churn.csv", type="string"),
        "model_type": Param(
            "gradient_boosting", type="string", enum=["logistic_regression", "random_forest", "gradient_boosting"]
        ),
        "min_f1": Param(0.48, type="number", minimum=0, maximum=1),
        "min_roc_auc": Param(0.70, type="number", minimum=0, maximum=1),
        "inject_failure": Param("none", type="string", enum=["none", "corrupt_data", "transient_error"]),
    },
)
def churn_training_pipeline():
    @task
    def validate_dataset() -> dict:
        import pandas as pd

        from training.train import file_md5, read_git_commit, validate_dataframe

        conf = _conf()
        rel = conf["dataset_path"]
        path = (PROJECT_ROOT / rel).resolve()
        if PROJECT_ROOT.resolve() not in path.parents or not path.is_file():
            raise AirflowFailException(f"dataset '{rel}' not found inside the project")

        df = pd.read_csv(path)
        if conf["inject_failure"] == "corrupt_data":
            log.warning("INJECTED FAILURE: simulating an upstream export that lost the label column")
            df = pd.concat([df.drop(columns=["churned"]), df.head(800).drop(columns=["churned"])])

        report = validate_dataframe(df)
        for c in report.checks:
            log.info("validation check %-22s %s  %s", c["check"], "PASS" if c["passed"] else "FAIL", c["detail"])
        if not report.passed:
            failed = "; ".join(f"{c['check']}: {c['detail']}" for c in report.failures)
            raise AirflowFailException(f"DATA_VALIDATION_FAILED: {failed}")

        info = {
            "dataset_path": rel,
            "dataset_md5": file_md5(path),
            "git_commit": read_git_commit(PROJECT_ROOT) or "unknown",
            "stats": report.stats,
        }
        log.info("dataset validated: %s", json.dumps(info))
        return info

    @task
    def preprocess_data(dataset: dict) -> dict:
        import pandas as pd

        from training.preprocess import preprocess
        from training.train import split

        ctx = get_current_context()
        if _conf()["inject_failure"] == "transient_error" and ctx["ti"].try_number == 1:
            log.warning("INJECTED FAILURE: simulating a transient storage timeout on the first attempt")
            raise TimeoutError("TRANSIENT: object storage read timed out (attempt 1)")

        df = preprocess(pd.read_csv(PROJECT_ROOT / dataset["dataset_path"]))
        train, test = split(df)
        wd = _workdir()
        train.to_csv(wd / "train.csv", index=False)
        test.to_csv(wd / "test.csv", index=False)
        log.info("preprocessed: train=%d test=%d features=%d", len(train), len(test), df.shape[1] - 1)
        return {"train": str(wd / "train.csv"), "test": str(wd / "test.csv"), "train_rows": len(train)}

    @task
    def train_model(splits: dict) -> dict:
        import joblib
        import pandas as pd

        from training.train import train_and_evaluate

        conf = _conf()
        train, test = pd.read_csv(splits["train"]), pd.read_csv(splits["test"])
        model, params, metrics = train_and_evaluate(train, test, conf["model_type"])
        path = _workdir() / "model.joblib"
        joblib.dump(model, path)
        log.info("trained %s params=%s", conf["model_type"], params)
        return {"model_path": str(path), "params": params, "metrics": metrics}

    @task
    def evaluate_and_log(dataset: dict, splits: dict, trained: dict) -> dict:
        import joblib
        import pandas as pd

        from training.train import log_to_mlflow

        ctx = get_current_context()
        model = joblib.load(trained["model_path"])
        train = pd.read_csv(splits["train"])
        tags = {
            "git_commit": dataset["git_commit"],
            "dataset_md5": dataset["dataset_md5"],
            "dataset_path": dataset["dataset_path"],
            "dataset_version": "unresolved",  # resolved from md5 by the agent's DVC tools
            "source": "airflow",
            "airflow_dag_id": ctx["dag"].dag_id,
            "airflow_run_id": ctx["dag_run"].run_id,
            "quality_gate": "pending",
        }
        out = log_to_mlflow(
            model,
            trained["params"],
            trained["metrics"],
            experiment=EXPERIMENT,
            tags=tags,
            train=train,
            run_name=f"airflow-{trained['params']['model_type']}",
            extra_artifacts={"validation_stats.json": dataset["stats"]},
        )
        log.info("logged to MLflow run %s metrics=%s", out["run_id"], trained["metrics"])
        return {**out, "metrics": trained["metrics"]}

    @task
    def quality_gate(logged: dict) -> dict:
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        from training.train import DEFAULT_GATE, REGISTERED_MODEL_NAME, gate_decision

        conf = _conf()
        gate = {**DEFAULT_GATE, "min_f1": float(conf["min_f1"]), "min_roc_auc": float(conf["min_roc_auc"])}
        client = MlflowClient()
        champion_f1 = None
        try:
            champ = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
            champion_f1 = client.get_run(champ.run_id).data.metrics.get("f1")
        except MlflowException:
            log.info("no champion model registered yet - absolute thresholds only")

        decision = gate_decision(logged["metrics"], champion_f1, gate)
        client.set_tag(logged["run_id"], "quality_gate", "passed" if decision["passed"] else "failed")
        client.set_tag(logged["run_id"], "quality_gate_reasons", "; ".join(decision["reasons"]) or "all checks passed")
        log.info("quality gate decision: %s", json.dumps(decision))
        if not decision["passed"]:
            raise AirflowFailException("QUALITY_GATE_FAILED: " + "; ".join(decision["reasons"]))
        return {**logged, "gate": decision}

    @task
    def register_candidate(gated: dict) -> dict:
        import mlflow
        from mlflow import MlflowClient

        from training.train import REGISTERED_MODEL_NAME

        mv = mlflow.register_model(gated["model_uri"], REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "candidate", mv.version)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "quality_gate", "passed")
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "source_run_id", gated["run_id"])
        client.update_model_version(
            REGISTERED_MODEL_NAME,
            mv.version,
            description="Passed automated quality gate. Awaiting human review before promotion to 'champion'.",
        )
        log.info("registered %s version %s with alias 'candidate'", REGISTERED_MODEL_NAME, mv.version)
        return {"model": REGISTERED_MODEL_NAME, "version": mv.version, "alias": "candidate"}

    dataset = validate_dataset()
    splits = preprocess_data(dataset)
    trained = train_model(splits)
    logged = evaluate_and_log(dataset, splits, trained)
    register_candidate(quality_gate(logged))


churn_training_pipeline()
