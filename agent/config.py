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

    # LLM (Gemini). The key is only ever read from the environment / .env and never logged.
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-flash-lite-latest"))
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"), repr=False)


def get_settings() -> Settings:
    return Settings()
