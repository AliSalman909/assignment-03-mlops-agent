from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent.config import Settings
from agent.tools.base import ToolRegistry
from agent.tools.dvc_tools import DVCTools, SafeCommandRunner, register_dvc_tools

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cwd: Path, *cmd: str) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _dvc(cwd: Path, *args: str) -> None:
    _run(cwd, sys.executable, "-m", "dvc", *args)


@pytest.fixture(scope="session")
def dvc_repo(tmp_path_factory) -> Path:
    """A throw-away git+DVC repo with two tagged dataset versions pushed to a local remote."""
    pytest.importorskip("dvc")
    root = tmp_path_factory.mktemp("repo")
    remote = tmp_path_factory.mktemp("remote")
    _run(root, "git", "init", "-b", "main")
    _run(root, "git", "config", "user.email", "test@example.com")
    _run(root, "git", "config", "user.name", "test")
    _dvc(root, "init")
    _dvc(root, "config", "core.analytics", "false")
    _dvc(root, "remote", "add", "-d", "storage", str(remote))
    (root / "data").mkdir()

    csv = root / "data" / "ds.csv"
    csv.write_text("a,b,churned\n1,2,0\n3,4,1\n", newline="\n")
    _dvc(root, "add", "data/ds.csv")
    _run(root, "git", "add", "-A")
    _run(root, "git", "commit", "-m", "dataset v1.0")
    _run(root, "git", "tag", "data-v1.0")
    _dvc(root, "push")

    csv.write_text("a,b,c,churned\n1,2,9,0\n3,4,9,1\n5,6,9,1\n", newline="\n")
    _dvc(root, "add", "data/ds.csv")
    _run(root, "git", "add", "-A")
    _run(root, "git", "commit", "-m", "dataset v2.0: add column c")
    _run(root, "git", "tag", "data-v2.0")
    _dvc(root, "push")
    return root


@pytest.fixture()
def dvc_registry(dvc_repo: Path, tmp_path: Path) -> tuple[ToolRegistry, DVCTools]:
    settings = Settings(repo_root=dvc_repo, default_dataset="data/ds.csv", reproduce_dir="artifacts/reproduced")
    tools = DVCTools(settings=settings, runner=SafeCommandRunner(dvc_repo, timeout_s=60))
    registry = ToolRegistry()
    register_dvc_tools(registry, tools)
    return registry, tools
