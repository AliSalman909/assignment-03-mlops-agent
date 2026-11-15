"""Runtime configuration, read from environment variables (never hard-coded secrets)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def get_settings() -> Settings:
    return Settings()
