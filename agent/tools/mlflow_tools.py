"""Task 2 - MLflow Experiment Intelligence module.

All access goes through `mlflow.MlflowClient` (never the web UI). Read-only: this module cannot
create runs, register models or move aliases - promotion is a human/pipeline decision.

Tools
  list_experiments        experiments with run counts
  list_training_runs             recent runs with metrics, params and provenance tags
  get_run_details         one run: metrics, params, tags, artifacts, registered versions
  find_best_model         best run by a metric + an explicit production-readiness assessment
  compare_model_runs      side-by-side comparison of given runs (or the last N) with a recommendation
  compare_experiments     aggregate comparison of two or more experiments
  get_model_versions      registered models, versions, aliases (champion/candidate) and source runs

Why the readiness assessment: the highest metric is *not* evidence of production readiness.
Every recommendation reports the margin over the runner-up, trade-offs on other metrics, the
quality-gate result, whether provenance (git commit + dataset md5) was recorded, and the
registry status, then states what still needs human review.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent.config import Settings, get_settings
from agent.tools.base import (
    Tool,
    ToolError,
    ToolInvalidRequestError,
    ToolOutput,
    ToolRegistry,
    ToolTimeoutError,
    ToolUnavailableError,
)

MODULE = "mlflow"
METRIC_DIRECTION = {"f1": "max", "precision": "max", "recall": "max", "accuracy": "max", "roc_auc": "max"}
KEY_METRICS = ["f1", "precision", "recall", "roc_auc", "accuracy"]
PROVENANCE_TAGS = ["git_commit", "dataset_md5", "dataset_path", "dataset_version", "source", "quality_gate"]
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-/]{1,120}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_METRIC_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


# ---------------------------------------------------------------------------- inputs
def _check_name(v: str) -> str:
    if not _NAME_RE.match(v):
        raise ValueError(f"'{v}' is not a valid MLflow name")
    return v


def _check_metric(v: str) -> str:
    if not _METRIC_RE.match(v):
        raise ValueError(f"'{v}' is not a valid metric name")
    return v


class ListExperimentsInput(BaseModel):
    include_deleted: bool = False


class ListRunsInput(BaseModel):
    experiment_id: str | None = Field(
        None, description="Experiment id (e.g. '1') or name (e.g. 'churn-baseline'); omit for all experiments"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum number of runs, newest first")
    filter_string: str | None = Field(
        None, description="Optional MLflow search filter, e.g. \"metrics.f1 > 0.5 and tags.source = 'airflow'\""
    )

    check_name = field_validator("experiment_id")(lambda cls, v: v if v is None else _check_name(v))

    @field_validator("filter_string")
    @classmethod
    def _v_filter(cls, v: str | None) -> str | None:
        if v is not None and (len(v) > 300 or ";" in v):
            raise ValueError("filter_string is too long or contains ';'")
        return v


class GetRunDetailsInput(BaseModel):
    run_id: str = Field(..., description="32-character MLflow run id")

    @field_validator("run_id")
    @classmethod
    def _v_run(cls, v: str) -> str:
        v = v.strip()
        if not _RUN_ID_RE.match(v):
            raise ValueError("run_id must be a 32-character hex MLflow run id")
        return v


class FindBestRunInput(BaseModel):
    experiment_id: str | None = Field(None, description="Experiment id or name; omit to search all experiments")
    metric: str = Field("f1", description="Evaluation metric to rank by")
    order: str | None = Field(
        None, description="'max'/'DESC' (higher is better) or 'min'/'ASC'; inferred for known metrics"
    )
    max_candidates: int = Field(50, ge=1, le=500)

    check_metric = field_validator("metric")(lambda cls, v: _check_metric(v))
    check_name = field_validator("experiment_id")(lambda cls, v: v if v is None else _check_name(v))

    @field_validator("order")
    @classmethod
    def _v_order(cls, v: str | None) -> str | None:
        if v is None:
            return v
        norm = {"max": "max", "desc": "max", "min": "min", "asc": "min"}.get(v.strip().lower())
        if norm is None:
            raise ValueError("order must be 'max'/'DESC' or 'min'/'ASC'")
        return norm


class CompareRunsInput(BaseModel):
    run_ids: list[str] | None = Field(None, description="Specific run ids to compare (2-20)")
    last_n: int = Field(5, ge=2, le=20, description="Used when run_ids is omitted: compare the N most recent runs")
    experiment_id: str | None = Field(None, description="Restrict last_n to one experiment (id or name)")
    metric: str = Field("f1", description="Metric used for the recommendation")

    check_metric = field_validator("metric")(lambda cls, v: _check_metric(v))

    @field_validator("run_ids")
    @classmethod
    def _v_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not 2 <= len(v) <= 20 or not all(_RUN_ID_RE.match(x) for x in v):
            raise ValueError("run_ids must be 2-20 MLflow run ids")
        return v


class CompareExperimentsInput(BaseModel):
    experiment_ids: list[str] = Field(..., min_length=2, max_length=10, description="Experiment ids or names")
    metric: str = Field("f1")

    check_metric = field_validator("metric")(lambda cls, v: _check_metric(v))

    @field_validator("experiment_ids")
    @classmethod
    def _v_names(cls, v: list[str]) -> list[str]:
        return [_check_name(n) for n in v]


class RegistryInfoInput(BaseModel):
    model_name: str | None = Field(None, description="Registered model name; omit to list all")

    check_name = field_validator("model_name")(lambda cls, v: v if v is None else _check_name(v))


# ---------------------------------------------------------------------------- helpers
def _ts(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat() if ms else None


class MLflowTools:
    def __init__(self, settings: Settings | None = None, client: Any = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            # fail fast instead of MLflow's default ~2 minutes of retries when the server is down
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(self.settings.mlflow_timeout_s)))
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "0")
            from mlflow import MlflowClient

            self._client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        return self._client

    def _call(self, fn, *args, **kwargs):
        """Invoke the MLflow client and translate failures into typed tool errors."""
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - classify below
            text = f"{type(exc).__name__}: {exc}"
            code = getattr(exc, "error_code", "")
            if code in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE") or "does not exist" in text.lower():
                raise ToolInvalidRequestError(f"MLflow: {str(exc)[:300]}") from exc
            if "timed out" in text.lower() or "timeout" in type(exc).__name__.lower():
                raise ToolTimeoutError(f"MLflow server at {self.settings.mlflow_tracking_uri} timed out") from exc
            if any(s in text for s in ("Connection refused", "Max retries", "ConnectionError", "Failed to establish")):
                raise ToolUnavailableError(
                    f"MLflow tracking server at {self.settings.mlflow_tracking_uri} is unreachable"
                ) from exc
            raise ToolError(f"MLflow error: {text[:300]}") from exc

    def _experiment_ids(self, names: list[str] | None) -> dict[str, str]:
        if not names:
            exps = self._call(self.client.search_experiments)
            return {e.experiment_id: e.name for e in exps}
        out = {}
        for n in names:  # each entry may be an experiment id ("25") or a name ("churn-baseline")
            exp = self._call(self.client.get_experiment, n) if n.isdigit() else None
            if exp is None:
                exp = self._call(self.client.get_experiment_by_name, n)
            if exp is None:
                raise ToolInvalidRequestError(f"experiment '{n}' does not exist")
            out[exp.experiment_id] = exp.name
        return out

    def _search(self, exp_ids: dict[str, str], max_results: int, filter_string: str | None = None, order=None):
        if not exp_ids:
            return []
        return self._call(
            self.client.search_runs,
            experiment_ids=list(exp_ids),
            filter_string=filter_string or "",
            order_by=order or ["attributes.start_time DESC"],
            max_results=max_results,
        )

    @staticmethod
    def _summary(run, exp_names: dict[str, str] | None = None) -> dict[str, Any]:
        tags = run.data.tags
        return {
            "run_id": run.info.run_id,
            "run_name": getattr(run.info, "run_name", None) or tags.get("mlflow.runName"),
            "experiment_id": run.info.experiment_id,
            "experiment_name": (exp_names or {}).get(run.info.experiment_id),
            "status": run.info.status,
            "start_time": _ts(run.info.start_time),
            "model_type": run.data.params.get("model_type"),
            "metrics": {k: v for k, v in run.data.metrics.items() if k in KEY_METRICS or k == "train_rows"},
            "provenance": {k: tags.get(k) for k in PROVENANCE_TAGS if tags.get(k) is not None},
        }

    def _versions_for_runs(self) -> dict[str, list[dict[str, Any]]]:
        """run_id -> registered model versions (with aliases) created from that run."""
        out: dict[str, list[dict[str, Any]]] = {}
        try:
            models = self._call(self.client.search_registered_models)
        except ToolInvalidRequestError:
            return out
        for m in models:
            aliases_by_version: dict[str, list[str]] = {}
            for alias, ver in (getattr(m, "aliases", None) or {}).items():
                aliases_by_version.setdefault(str(ver), []).append(alias)
            for mv in self._call(self.client.search_model_versions, f"name='{m.name}'"):
                out.setdefault(mv.run_id, []).append(
                    {
                        "model": m.name,
                        "version": str(mv.version),
                        "aliases": aliases_by_version.get(str(mv.version), []),
                    }
                )
        return out

    def _readiness(self, ranked: list[dict[str, Any]], metric: str, mode: str) -> dict[str, Any]:
        """Explain why the top run is (or is not yet) a defensible production candidate."""
        best = ranked[0]
        versions = self._versions_for_runs()
        findings: list[str] = []
        blockers: list[str] = []
        val = best["metrics"].get(metric)

        if len(ranked) > 1 and ranked[1]["metrics"].get(metric) is not None:
            margin = abs(val - ranked[1]["metrics"][metric])
            if margin < 0.01:
                findings.append(
                    f"margin over runner-up ({ranked[1]['run_id'][:8]}) is only {margin:.4f} {metric}: "
                    "within typical evaluation noise, so the ranking is not decisive"
                )
            else:
                findings.append(f"leads runner-up by {margin:.4f} {metric}")
        m = best["metrics"]
        if "precision" in m and "recall" in m and abs(m["precision"] - m["recall"]) > 0.15:
            findings.append(
                f"precision ({m['precision']:.3f}) and recall ({m['recall']:.3f}) are unbalanced; "
                "check which error type the business cares about"
            )
        for other in ("roc_auc", "f1", "recall", "precision"):
            if other == metric or other not in m:
                continue
            best_other = max((r["metrics"].get(other, float("-inf")) for r in ranked), default=None)
            if best_other is not None and m[other] < best_other - 0.02:
                findings.append(f"not the best on {other} ({m[other]:.3f} vs best {best_other:.3f})")

        prov = best["provenance"]
        if not (prov.get("git_commit") and prov.get("dataset_md5")):
            blockers.append("no dataset provenance (git_commit/dataset_md5) recorded - training data unverifiable")
        gate = prov.get("quality_gate")
        if gate == "failed":
            blockers.append("this run FAILED the automated quality gate")
        elif gate != "passed":
            findings.append("no automated quality-gate result recorded for this run")
        regs = versions.get(best["run_id"], [])
        if not regs:
            findings.append("not registered in the model registry")
        if best["status"] != "FINISHED":
            blockers.append(f"run status is {best['status']}")

        return {
            "recommended_run_id": best["run_id"],
            "selection_basis": f"{mode} {metric} = {val}",
            "registered_versions": regs,
            "observations": findings,
            "blockers": blockers,
            "production_ready": "no - blockers present" if blockers else "not determined by metrics alone",
            "still_requires": [
                "human review and explicit promotion to the 'champion' alias",
                "confirmation that it was trained on the current approved dataset version",
                "checks not recorded in MLflow (latency, robustness, fairness, business sign-off)",
            ],
        }

    # ------------------------------------------------------------------ tools
    def list_experiments(self, p: ListExperimentsInput) -> ToolOutput:
        from mlflow.entities import ViewType

        view = ViewType.ALL if p.include_deleted else ViewType.ACTIVE_ONLY
        exps = self._call(self.client.search_experiments, view_type=view)
        rows = []
        for e in exps:
            runs = self._call(self.client.search_runs, experiment_ids=[e.experiment_id], max_results=1000)
            rows.append(
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "lifecycle_stage": e.lifecycle_stage,
                    "num_runs": len(runs),
                    "last_update_time": _ts(getattr(e, "last_update_time", None)),
                }
            )
        return ToolOutput(
            data={"tracking_uri": self.settings.mlflow_tracking_uri, "count": len(rows), "experiments": rows},
            evidence_source=["MlflowClient.search_experiments", "MlflowClient.search_runs"],
        )

    def list_runs(self, p: ListRunsInput) -> ToolOutput:
        exps = self._experiment_ids([p.experiment_id] if p.experiment_id else None)
        runs = self._search(exps, p.limit, p.filter_string)
        return ToolOutput(
            data={"count": len(runs), "runs": [self._summary(r, exps) for r in runs]},
            evidence_source=[f"MlflowClient.search_runs(experiments={list(exps.values())})"],
        )

    def get_run_details(self, p: GetRunDetailsInput) -> ToolOutput:
        run = self._call(self.client.get_run, p.run_id)
        exp = self._call(self.client.get_experiment, run.info.experiment_id)
        artifacts: list[dict[str, Any]] = []

        def walk(path: str | None, depth: int = 0) -> None:
            if depth > 2 or len(artifacts) >= 50:
                return
            for a in self._call(self.client.list_artifacts, p.run_id, path):
                artifacts.append({"path": a.path, "is_dir": a.is_dir, "size_bytes": a.file_size})
                if a.is_dir:
                    walk(a.path, depth + 1)

        walk(None)
        logged_models = []
        try:
            for lm in self._call(self.client.search_logged_models, experiment_ids=[run.info.experiment_id]):
                if getattr(lm, "source_run_id", None) == p.run_id:
                    logged_models.append(
                        {"model_id": lm.model_id, "name": lm.name, "model_uri": f"models:/{lm.model_id}"}
                    )
        except (ToolError, AttributeError):
            pass
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.") or k == "mlflow.runName"}
        return ToolOutput(
            data={
                **self._summary(run, {exp.experiment_id: exp.name}),
                "end_time": _ts(run.info.end_time),
                "params": dict(run.data.params),
                "all_metrics": dict(run.data.metrics),
                "tags": tags,
                "artifacts": artifacts,
                "logged_models": logged_models,
                "registered_versions": self._versions_for_runs().get(p.run_id, []),
            },
            evidence_source=[
                "MlflowClient.get_run",
                "MlflowClient.list_artifacts",
                "MlflowClient.search_model_versions",
            ],
        )

    def find_best_run(self, p: FindBestRunInput) -> ToolOutput:
        mode = p.order or METRIC_DIRECTION.get(p.metric, "max")
        exps = self._experiment_ids([p.experiment_id] if p.experiment_id else None)
        order = [f"metrics.`{p.metric}` {'DESC' if mode == 'max' else 'ASC'}"]
        runs = self._search(exps, p.max_candidates, "attributes.status = 'FINISHED'", order)
        ranked = [self._summary(r, exps) for r in runs if p.metric in r.data.metrics]
        if not ranked:
            raise ToolInvalidRequestError(f"no finished runs have recorded metric '{p.metric}'")
        return ToolOutput(
            data={
                "metric": p.metric,
                "mode": mode,
                "candidates_considered": len(ranked),
                "best_run": ranked[0],
                "top_5": [
                    {
                        "run_id": r["run_id"],
                        "run_name": r["run_name"],
                        "model_type": r["model_type"],
                        p.metric: r["metrics"][p.metric],
                        "dataset_md5": r["provenance"].get("dataset_md5"),
                    }
                    for r in ranked[:5]
                ],
                "assessment": self._readiness(ranked, p.metric, mode),
            },
            evidence_source=[f"MlflowClient.search_runs(order_by={order[0]})", "MlflowClient.search_model_versions"],
        )

    def compare_runs(self, p: CompareRunsInput) -> ToolOutput:
        if p.run_ids:
            runs = [self._call(self.client.get_run, rid) for rid in p.run_ids]
            exps = {
                r.info.experiment_id: self._call(self.client.get_experiment, r.info.experiment_id).name for r in runs
            }
            basis = "explicit run ids"
        else:
            exps = self._experiment_ids([p.experiment_id] if p.experiment_id else None)
            runs = self._search(exps, p.last_n, "attributes.status = 'FINISHED'")
            basis = f"{len(runs)} most recent finished runs"
        rows = [self._summary(r, exps) for r in runs]
        if len(rows) < 2:
            raise ToolInvalidRequestError("need at least two runs to compare")
        best_per_metric = {}
        for m in KEY_METRICS:
            vals = [(r["metrics"][m], r["run_id"]) for r in rows if m in r["metrics"]]
            if vals:
                v, rid = max(vals)
                best_per_metric[m] = {"value": v, "run_id": rid}
        mode = METRIC_DIRECTION.get(p.metric, "max")
        ranked = sorted(
            [r for r in rows if p.metric in r["metrics"]],
            key=lambda r: r["metrics"][p.metric],
            reverse=(mode == "max"),
        )
        if not ranked:
            raise ToolInvalidRequestError(f"none of the compared runs recorded metric '{p.metric}'")
        return ToolOutput(
            data={
                "basis": basis,
                "metric": p.metric,
                "runs": rows,
                "ranking": [{"run_id": r["run_id"], p.metric: r["metrics"][p.metric]} for r in ranked],
                "best_per_metric": best_per_metric,
                "recommendation": self._readiness(ranked, p.metric, mode),
            },
            evidence_source=["MlflowClient.search_runs/get_run"],
        )

    def compare_experiments(self, p: CompareExperimentsInput) -> ToolOutput:
        import statistics

        exps = self._experiment_ids(p.experiment_ids)
        summary = []
        for eid, name in exps.items():
            runs = self._search({eid: name}, 500, "attributes.status = 'FINISHED'")
            vals = [r.data.metrics[p.metric] for r in runs if p.metric in r.data.metrics]
            best = max(runs, key=lambda r: r.data.metrics.get(p.metric, float("-inf")), default=None)
            summary.append(
                {
                    "experiment": name,
                    "experiment_id": eid,
                    "finished_runs": len(runs),
                    "runs_with_metric": len(vals),
                    f"best_{p.metric}": max(vals) if vals else None,
                    f"mean_{p.metric}": round(statistics.mean(vals), 4) if vals else None,
                    f"std_{p.metric}": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
                    "best_run": self._summary(best, exps) if best and vals else None,
                    "latest_run_start": _ts(runs[0].info.start_time) if runs else None,
                    "model_types": sorted({r.data.params.get("model_type") for r in runs} - {None}),
                }
            )
        scored = [s for s in summary if s[f"best_{p.metric}"] is not None]
        leader = max(scored, key=lambda s: s[f"best_{p.metric}"]) if scored else None
        return ToolOutput(
            data={
                "metric": p.metric,
                "experiments": summary,
                "leader": leader["experiment"] if leader else None,
                "note": "experiments may use different dataset versions; compare provenance before concluding",
            },
            evidence_source=["MlflowClient.get_experiment_by_name", "MlflowClient.search_runs"],
        )

    def get_model_registry_info(self, p: RegistryInfoInput) -> ToolOutput:
        if p.model_name:
            models = [self._call(self.client.get_registered_model, p.model_name)]
        else:
            models = self._call(self.client.search_registered_models)
        out = []
        for m in models:
            aliases = {a: str(v) for a, v in (getattr(m, "aliases", None) or {}).items()}
            versions = []
            for mv in self._call(self.client.search_model_versions, f"name='{m.name}'"):
                try:
                    run_metrics = self._call(self.client.get_run, mv.run_id).data.metrics if mv.run_id else {}
                    run_tags = self._call(self.client.get_run, mv.run_id).data.tags if mv.run_id else {}
                except ToolInvalidRequestError:
                    run_metrics, run_tags = {}, {}
                versions.append(
                    {
                        "version": str(mv.version),
                        "aliases": [a for a, v in aliases.items() if v == str(mv.version)],
                        "status": mv.status,
                        "source_run_id": mv.run_id,
                        "created": _ts(mv.creation_timestamp),
                        "description": mv.description,
                        "tags": dict(mv.tags or {}),
                        "metrics": {k: v for k, v in run_metrics.items() if k in KEY_METRICS},
                        "provenance": {k: run_tags.get(k) for k in PROVENANCE_TAGS if run_tags.get(k)},
                    }
                )
            versions.sort(key=lambda v: int(v["version"]), reverse=True)
            out.append(
                {
                    "name": m.name,
                    "aliases": aliases,
                    "latest_version": versions[0]["version"] if versions else None,
                    "versions": versions,
                }
            )
        return ToolOutput(
            data={"count": len(out), "registered_models": out},
            evidence_source=["MlflowClient.search_registered_models", "MlflowClient.search_model_versions"],
        )


def register_mlflow_tools(registry: ToolRegistry, tools: MLflowTools | None = None) -> MLflowTools:
    t = tools or MLflowTools()
    specs = [
        (
            "list_experiments",
            ListExperimentsInput,
            t.list_experiments,
            "List MLflow experiments with their number of runs.",
        ),
        (
            "list_training_runs",
            ListRunsInput,
            t.list_runs,
            "List recent MLflow training runs (newest first) with key metrics, model type and provenance tags "
            "(git_commit, dataset_md5, quality_gate). Optionally restrict to one experiment or a filter.",
        ),
        (
            "get_run_details",
            GetRunDetailsInput,
            t.get_run_details,
            "Get one run's full metrics, parameters, tags, artifact list, logged models and registered versions.",
        ),
        (
            "find_best_model",
            FindBestRunInput,
            t.find_best_run,
            "Find the best finished run by an evaluation metric (default f1) and assess whether it is a defensible "
            "production candidate (margin, trade-offs, quality gate, provenance, registry). Highest metric alone is "
            "never treated as production-ready.",
        ),
        (
            "compare_model_runs",
            CompareRunsInput,
            t.compare_runs,
            "Compare specific runs, or the last N finished runs, side by side with best-per-metric and a reasoned "
            "recommendation including caveats.",
        ),
        (
            "compare_experiments",
            CompareExperimentsInput,
            t.compare_experiments,
            "Compare two or more experiments: run counts, best/mean/std of a metric, best run and model types.",
        ),
        (
            "get_model_versions",
            RegistryInfoInput,
            t.get_model_registry_info,
            "Model registry: registered models, versions, aliases (champion = serving, candidate = awaiting review), "
            "source runs, their metrics and dataset provenance.",
        ),
    ]
    for name, model, func, desc in specs:
        registry.register(Tool(name=name, module=MODULE, description=desc, input_model=model, func=func))
    return t
