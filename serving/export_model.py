"""Export a registered model (default: churn-classifier@champion) into serving/model/ for the image.

Writes the MLflow model directory plus model_meta.json with full lineage:
registry name/version/alias, source run, metrics, decision threshold, git commit, dataset md5,
and the dataset version resolved through the agent's DVC tool (not trusted from a tag).

    python serving/export_model.py --alias champion
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "serving" / "model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="churn-classifier")
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    import mlflow
    from mlflow import MlflowClient

    from agent.tools.base import ToolRegistry
    from agent.tools.dvc_tools import register_dvc_tools

    client = MlflowClient()
    mv = client.get_model_version_by_alias(args.model_name, args.alias)
    run = client.get_run(mv.run_id)

    # verify provenance BEFORE touching serving/model, so a refused export leaves the previous model intact
    tags = run.data.tags
    if not (tags.get("git_commit") and tags.get("dataset_md5")):
        raise SystemExit("refusing to export: the source run has no git_commit/dataset_md5 provenance tags")
    reg = ToolRegistry()
    register_dvc_tools(reg)
    ds = reg.execute("get_dataset_version", {"git_commit": tags["git_commit"], "expected_md5": tags["dataset_md5"]})
    if not ds.ok or not ds.data.get("md5_matches_expected"):
        raise SystemExit(f"refusing to export: dataset provenance could not be verified ({ds.error})")

    tmp = Path(mlflow.artifacts.download_artifacts(f"models:/{args.model_name}/{mv.version}"))
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(tmp, OUT / "mlflow_model")

    meta = {
        "model_name": args.model_name,
        "model_version": str(mv.version),
        "model_alias": args.alias,
        "mlflow_run_id": mv.run_id,
        "model_type": run.data.params.get("model_type"),
        "decision_threshold": float(run.data.params.get("decision_threshold", 0.5)),
        "metrics": {k: v for k, v in run.data.metrics.items() if k in ("f1", "precision", "recall", "roc_auc")},
        "git_commit": tags["git_commit"],
        "dataset_md5": tags["dataset_md5"],
        "dataset_version": ds.data["version"],
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
