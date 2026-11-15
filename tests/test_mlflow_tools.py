"""Tests for the MLflow experiment-intelligence module (Task 2) against a fake MlflowClient."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from agent.config import Settings
from agent.schemas import ToolStatus
from agent.tools.base import ToolRegistry
from agent.tools.mlflow_tools import MLflowTools, register_mlflow_tools


def _run(
    rid: str,
    exp: str,
    f1: float,
    *,
    gate: str | None = "passed",
    status="FINISHED",
    start=0,
    prov=True,
    precision=0.5,
    recall=0.5,
    auc=0.7,
):
    tags = {"mlflow.runName": f"run-{rid[:4]}"}
    if prov:
        tags |= {"git_commit": "a" * 40, "dataset_md5": "b" * 32, "dataset_version": "v2.0"}
    if gate:
        tags["quality_gate"] = gate
    return NS(
        info=NS(
            run_id=rid,
            run_name=f"run-{rid[:4]}",
            experiment_id=exp,
            status=status,
            start_time=start,
            end_time=start + 1,
        ),
        data=NS(
            metrics={"f1": f1, "precision": precision, "recall": recall, "roc_auc": auc},
            params={"model_type": "gradient_boosting"},
            tags=tags,
        ),
    )


class FakeClient:
    def __init__(self, runs, experiments, models=None):
        self.runs = {r.info.run_id: r for r in runs}
        self.experiments = experiments  # id -> name
        self.models = models or []

    def search_experiments(self, view_type=None):
        return [
            NS(experiment_id=i, name=n, lifecycle_stage="active", last_update_time=0)
            for i, n in self.experiments.items()
        ]

    def get_experiment_by_name(self, name):
        return next((e for e in self.search_experiments() if e.name == name), None)

    def get_experiment(self, eid):
        return NS(experiment_id=eid, name=self.experiments[eid])

    def search_runs(self, experiment_ids, filter_string="", order_by=None, max_results=100):
        rows = [r for r in self.runs.values() if r.info.experiment_id in experiment_ids]
        if "FINISHED" in (filter_string or ""):
            rows = [r for r in rows if r.info.status == "FINISHED"]
        if order_by and "metrics." in order_by[0]:
            rows.sort(key=lambda r: r.data.metrics.get("f1", -1), reverse="DESC" in order_by[0])
        else:
            rows.sort(key=lambda r: r.info.start_time, reverse=True)
        return rows[:max_results]

    def get_run(self, rid):
        if rid not in self.runs:
            raise type("MlflowException", (Exception,), {"error_code": "RESOURCE_DOES_NOT_EXIST"})(
                f"Run {rid} not found"
            )
        return self.runs[rid]

    def list_artifacts(self, rid, path=None):
        return [NS(path="model", is_dir=False, file_size=10)] if path is None else []

    def search_logged_models(self, experiment_ids):
        return []

    def search_registered_models(self):
        return [NS(name=m["name"], aliases=m["aliases"]) for m in self.models]

    def get_registered_model(self, name):
        return next(NS(name=m["name"], aliases=m["aliases"]) for m in self.models if m["name"] == name)

    def search_model_versions(self, filter_string):
        name = filter_string.split("'")[1]
        m = next(m for m in self.models if m["name"] == name)
        return [
            NS(version=v, run_id=rid, status="READY", creation_timestamp=0, description="", tags={})
            for v, rid in m["versions"].items()
        ]


RID = [f"{i:032x}" for i in range(1, 8)]


@pytest.fixture()
def registry():
    runs = [
        _run(RID[0], "1", 0.47, start=1),
        _run(RID[1], "1", 0.51, start=2, gate=None),
        _run(RID[2], "1", 0.505, start=3, precision=0.43, recall=0.61),
        _run(RID[3], "2", 0.52, start=4, gate="failed"),
        _run(RID[4], "2", 0.30, start=5, status="FAILED"),
    ]
    models = [
        {
            "name": "churn-classifier",
            "aliases": {"champion": "1", "candidate": "2"},
            "versions": {"1": RID[0], "2": RID[2]},
        }
    ]
    client = FakeClient(runs, {"1": "churn-baseline", "2": "churn-training-pipeline"}, models)
    reg = ToolRegistry()
    register_mlflow_tools(reg, MLflowTools(settings=Settings(), client=client))
    return reg


def test_find_best_run_does_not_equate_top_metric_with_production_ready(registry):
    res = registry.execute("find_best_run", {"metric": "f1"})
    assert res.ok, res.error
    assert res.data["best_run"]["run_id"] == RID[3]  # highest f1 ...
    a = res.data["assessment"]
    assert any("FAILED the automated quality gate" in b for b in a["blockers"])  # ... but failed the gate
    assert a["production_ready"].startswith("no")
    assert "human review" in " ".join(a["still_requires"])


def test_find_best_run_flags_indecisive_margin(registry):
    res = registry.execute("find_best_run", {"metric": "f1", "experiment_names": ["churn-baseline"]})
    assert res.ok
    assert res.data["best_run"]["run_id"] == RID[1]
    obs = " ".join(res.data["assessment"]["observations"])
    assert "not decisive" in obs  # 0.51 vs 0.505
    assert "no automated quality-gate result" in obs


def test_compare_last_n_runs_ranks_and_recommends(registry):
    res = registry.execute("compare_runs", {"last_n": 3, "experiment_name": "churn-baseline", "metric": "f1"})
    assert res.ok, res.error
    assert [r["run_id"] for r in res.data["ranking"]] == [RID[1], RID[2], RID[0]]
    assert res.data["best_per_metric"]["f1"]["run_id"] == RID[1]
    assert res.data["recommendation"]["production_ready"] != "yes"


def test_compare_experiments_aggregates(registry):
    res = registry.execute("compare_experiments", {"experiment_names": ["churn-baseline", "churn-training-pipeline"]})
    assert res.ok, res.error
    by = {e["experiment"]: e for e in res.data["experiments"]}
    assert by["churn-baseline"]["finished_runs"] == 3
    assert by["churn-training-pipeline"]["finished_runs"] == 1  # FAILED run excluded
    assert res.data["leader"] == "churn-training-pipeline"


def test_registry_info_reports_aliases_and_provenance(registry):
    res = registry.execute("get_model_registry_info", {"model_name": "churn-classifier"})
    assert res.ok, res.error
    model = res.data["registered_models"][0]
    assert model["aliases"] == {"champion": "1", "candidate": "2"}
    v1 = next(v for v in model["versions"] if v["version"] == "1")
    assert v1["aliases"] == ["champion"] and v1["provenance"]["dataset_version"] == "v2.0"


def test_run_details_and_unknown_run(registry):
    ok = registry.execute("get_run_details", {"run_id": RID[2]})
    assert ok.ok and ok.data["registered_versions"][0]["aliases"] == ["candidate"]
    missing = registry.execute("get_run_details", {"run_id": "f" * 32})
    assert missing.status == ToolStatus.INVALID_REQUEST and missing.data is None


def test_unknown_experiment_and_bad_input(registry):
    assert registry.execute("list_runs", {"experiment_name": "nope"}).status == ToolStatus.INVALID_REQUEST
    assert registry.execute("get_run_details", {"run_id": "../../etc"}).status == ToolStatus.INVALID_REQUEST
    assert registry.execute("find_best_run", {"metric": "f1; drop"}).status == ToolStatus.INVALID_REQUEST


def test_mlflow_server_down_is_unavailable_not_invented():
    class Down:
        def __getattr__(self, name):
            def boom(*a, **k):
                raise ConnectionError("Failed to establish a new connection: [Errno 111] Connection refused")

            return boom

    reg = ToolRegistry()
    register_mlflow_tools(reg, MLflowTools(settings=Settings(), client=Down()))
    res = reg.execute("find_best_run", {})
    assert res.status == ToolStatus.UNAVAILABLE
    assert res.data is None
