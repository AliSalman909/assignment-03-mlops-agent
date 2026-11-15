"""Runtime configuration, read from environment variables (never hard-coded secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # load a local, git-ignored .env if python-dotenv is installed; real env vars take precedence
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # DVC
    repo_root: Path = field(default_factory=lambda: Path(_env("MLOPS_REPO_ROOT", str(PROJECT_ROOT))).resolve())
    dvc_timeout_s: float = field(default_factory=lambda: float(_env("DVC_TIMEOUT_S", "60")))
    reproduce_dir: str = field(default_factory=lambda: _env("DVC_REPRODUCE_DIR", "artifacts/reproduced"))
    default_dataset: str = field(default_factory=lambda: _env("DVC_DEFAULT_DATASET", "data/raw/churn.csv"))
    dataset_tag_prefix: str = field(default_factory=lambda: _env("DVC_DATASET_TAG_PREFIX", "data-"))

    # MLflow
    mlflow_tracking_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow_timeout_s: float = field(default_factory=lambda: float(_env("MLFLOW_TIMEOUT_S", "10")))

    # Airflow 3 REST API
    airflow_base_url: str = field(default_factory=lambda: _env("AIRFLOW_BASE_URL", "http://localhost:8080"))
    airflow_username: str = field(default_factory=lambda: _env("AIRFLOW_USERNAME", "admin"))
    airflow_password: str | None = field(
        default_factory=lambda: os.environ.get("AIRFLOW_PASSWORD") or os.environ.get("AIRFLOW_ADMIN_PASSWORD"),
        repr=False,
    )
    airflow_timeout_s: float = field(default_factory=lambda: float(_env("AIRFLOW_TIMEOUT_S", "15")))
    airflow_approved_dags: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            d.strip() for d in _env("AIRFLOW_APPROVED_DAGS", "churn_training_pipeline").split(",") if d.strip()
        )
    )

    # Kubernetes
    k8s_kubeconfig: str | None = field(default_factory=lambda: os.environ.get("K8S_KUBECONFIG") or None)
    k8s_context: str | None = field(default_factory=lambda: os.environ.get("K8S_CONTEXT") or None)
    k8s_default_namespace: str = field(default_factory=lambda: _env("K8S_NAMESPACE", "mlops"))
    k8s_approved_namespaces: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            n.strip() for n in _env("K8S_APPROVED_NAMESPACES", "mlops").split(",") if n.strip()
        )
    )
    # only Deployments carrying this label may be modified (restart/scale/rollback)
    k8s_managed_label: str = field(
        default_factory=lambda: _env("K8S_MANAGED_LABEL", "app.kubernetes.io/part-of=mlops-agent")
    )
    k8s_timeout_s: float = field(default_factory=lambda: float(_env("K8S_TIMEOUT_S", "10")))
    k8s_max_replicas: int = field(default_factory=lambda: int(_env("K8S_MAX_REPLICAS", "5")))

    # LLM (Gemini). The key is only ever read from the environment / .env and never logged.
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-flash-lite-latest"))
    # "*-latest" aliases are only served on v1beta; pinned models (e.g. gemini-2.5-flash-lite) also work on v1
    gemini_api_version: str = field(default_factory=lambda: _env("GEMINI_API_VERSION", "v1beta"))
    gemini_temperature: float = field(default_factory=lambda: float(_env("GEMINI_TEMPERATURE", "0.1")))
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"), repr=False)
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "90")))

    # Agent loop budgets
    agent_max_steps: int = field(default_factory=lambda: int(_env("AGENT_MAX_STEPS", "10")))
    agent_max_tool_calls: int = field(default_factory=lambda: int(_env("AGENT_MAX_TOOL_CALLS", "25")))
    agent_tool_result_chars: int = field(default_factory=lambda: int(_env("AGENT_TOOL_RESULT_CHARS", "9000")))
    pending_action_ttl_s: int = field(default_factory=lambda: int(_env("PENDING_ACTION_TTL_S", "900")))

    # Retraining policy (explicit, so recommendations are evidence-based)
    retrain_max_model_age_days: int = field(default_factory=lambda: int(_env("RETRAIN_MAX_MODEL_AGE_DAYS", "30")))
    retrain_min_f1: float = field(default_factory=lambda: float(_env("RETRAIN_MIN_PRODUCTION_F1", "0.45")))

    # API
    agent_api_key: str | None = field(default_factory=lambda: os.environ.get("AGENT_API_KEY") or None, repr=False)


def get_settings() -> Settings:
    return Settings()
