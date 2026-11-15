"""Explicit retraining policy, exposed as a tool so recommendations cite a written rule.

The assignment requires that retraining is recommended only with evidence such as an explicit
freshness policy, a failed quality gate, or supplied degradation results. This module is that
explicit policy; thresholds come from configuration.
"""

from __future__ import annotations

from pydantic import BaseModel

from agent.config import Settings, get_settings
from agent.tools.base import Tool, ToolOutput, ToolRegistry

MODULE = "policy"


class PolicyInput(BaseModel):
    pass


def retraining_policy(settings: Settings) -> dict:
    return {
        "policy_id": "churn-retraining-policy-v1",
        "applies_to": "registered model 'churn-classifier', alias 'champion' (the model serving production traffic)",
        "rules": [
            {
                "id": "DATA_FRESHNESS",
                "trigger": "a newer tagged dataset version exists than the one the champion model was trained on",
                "how_to_check": "compare the champion's dataset version (get_model_versions provenance -> "
                "get_dataset_version with git_commit + expected_md5) against the latest version from "
                "get_dataset_history",
                "action": "retraining on the latest data is warranted; if a 'candidate' version already trained on "
                "the latest data passed the quality gate, recommend human review/promotion of that candidate instead",
            },
            {
                "id": "MODEL_AGE",
                "trigger": f"the champion model version is older than {settings.retrain_max_model_age_days} days",
                "how_to_check": "registry 'created' timestamp of the champion version",
                "action": "schedule retraining",
            },
            {
                "id": "QUALITY_GATE_FAILURE",
                "trigger": "the most recent training pipeline run failed its quality gate",
                "how_to_check": "get_dag_runs + diagnose_dag_run (failure layer 'model')",
                "action": "do NOT promote; investigate data/features before any retraining",
            },
            {
                "id": "PRODUCTION_DEGRADATION",
                "trigger": f"supplied production evaluation shows F1 below {settings.retrain_min_f1}",
                "how_to_check": "only from results explicitly supplied by the user; none are collected automatically",
                "action": "retraining warranted",
            },
        ],
        "not_sufficient_on_their_own": [
            "a different run having a slightly higher offline metric",
            "failed pipeline runs caused by data validation or infrastructure (fix those first)",
            "the absence of information (unknown is not evidence)",
        ],
        "execution": "retraining is only started by an explicit user request and confirmation of "
        "trigger_training_pipeline; a recommendation never triggers it automatically",
    }


def register_policy_tools(registry: ToolRegistry, settings: Settings | None = None) -> None:
    s = settings or get_settings()
    registry.register(
        Tool(
            name="get_retraining_policy",
            module=MODULE,
            description="Return the organisation's explicit retraining policy (data freshness, model age, quality-gate "
            "failure, production degradation). Consult it before recommending for or against retraining.",
            input_model=PolicyInput,
            func=lambda p: ToolOutput(data=retraining_policy(s), evidence_source=["agent/tools/policy_tools.py"]),
        )
    )
