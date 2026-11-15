"""Tests for the DVC data-management module (Task 1)."""

from __future__ import annotations

import logging
import subprocess

import pytest

from agent.config import Settings
from agent.schemas import ToolStatus
from agent.tools.base import ToolRegistry
from agent.tools.dvc_tools import (
    DVCTools,
    SafeCommandRunner,
    normalize_path,
    register_dvc_tools,
    validate_rev,
)


# ---------------------------------------------------------------- real repo ---
def test_list_datasets_finds_tracked_file(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute("list_datasets", {"git_commit": "HEAD"})
    assert res.status == ToolStatus.OK, res.error
    paths = [d["path"] for d in res.data["datasets"]]
    assert "data/ds.csv" in paths


def test_dataset_version_identified_from_tag_and_md5(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute("get_dataset_version", {"dataset_path": "data/ds.csv", "git_commit": "data-v1.0"})
    assert res.status == ToolStatus.OK, res.error
    assert res.data["version"] == "v1.0"
    assert len(res.data["commit"]) == 40
    assert res.data["md5"] and len(res.data["md5"]) == 32
    assert res.data["default_remote"] == "storage"


def test_provenance_md5_verification(dvc_registry):
    registry, _ = dvc_registry
    v1 = registry.execute("get_dataset_version", {"dataset_path": "data/ds.csv", "git_commit": "data-v1.0"}).data["md5"]
    res = registry.execute(
        "get_dataset_version", {"dataset_path": "data/ds.csv", "git_commit": "data-v2.0", "expected_md5": v1}
    )
    assert res.ok and res.data["md5_matches_expected"] is False


def test_history_lists_both_versions(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute("get_dataset_history", {"dataset_path": "data/ds.csv"})
    assert res.ok, res.error
    tags = [t for v in res.data["versions"] for t in v["tags"]]
    assert {"data-v1.0", "data-v2.0"} <= set(tags)


def test_compare_versions_reports_schema_and_row_changes(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute(
        "compare_dataset_versions", {"dataset_path": "data/ds.csv", "version_a": "data-v1.0", "version_b": "data-v2.0"}
    )
    assert res.ok, res.error
    assert res.data["content_identical"] is False
    assert res.data["content"]["row_delta"] == 1
    assert res.data["content"]["columns_added"] == ["c"]


def test_detects_modified_data_and_restore_requires_confirmation(dvc_registry):
    registry, tools = dvc_registry
    csv = tools.root / "data" / "ds.csv"
    original = csv.read_bytes()
    try:
        assert registry.execute("check_dataset_changes", {"check_remote": False}).data["data_changed"] is False
        csv.write_text("tampered\n")
        res = registry.execute("check_dataset_changes", {"check_remote": False})
        assert res.ok and res.data["data_changed"] is True

        refused = registry.execute("restore_dataset_workspace", {"dataset_path": "data/ds.csv"})
        assert refused.status == ToolStatus.CONFIRMATION_REQUIRED
        assert csv.read_text() == "tampered\n"  # nothing executed

        ok = registry.execute("restore_dataset_workspace", {"dataset_path": "data/ds.csv"}, confirmed=True)
        assert ok.ok and ok.data["verified"] is True
    finally:
        csv.write_bytes(original)


def test_materialize_previous_version_is_byte_identical(dvc_registry):
    registry, tools = dvc_registry
    res = registry.execute("reproduce_dataset", {"dataset_path": "data/ds.csv", "git_commit": "data-v1.0"})
    assert res.ok, res.error
    assert res.data["verified"] is True
    assert res.data["reproduced_md5"] == res.data["recorded_md5"]
    assert (tools.root / res.data["output_file"]).read_text().startswith("a,b,churned")


def test_reproduce_instructions_contain_exact_commands(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute("explain_dataset_reproduction", {"dataset_path": "data/ds.csv", "git_commit": "data-v1.0"})
    assert res.ok, res.error
    commands = " ".join(s["command"] for s in res.data["steps"])
    assert "git checkout" in commands and "dvc pull data/ds.csv.dvc" in commands
    assert res.data["availability"]["in_default_remote"] is True


def test_version_labels_resolve_to_tags(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute(
        "compare_dataset_versions", {"dataset_path": "data/ds.csv", "version_a": "v1.0", "version_b": "v2.0"}
    )
    assert res.ok, res.error
    assert res.data["a"]["version"] == "v1.0" and res.data["b"]["version"] == "v2.0"


def test_unknown_revision_is_reported_not_invented(dvc_registry):
    registry, _ = dvc_registry
    res = registry.execute("get_dataset_version", {"dataset_path": "data/ds.csv", "git_commit": "data-v9.9"})
    assert res.status == ToolStatus.INVALID_REQUEST
    assert res.data is None


# ------------------------------------------------------------ validation -----
@pytest.mark.parametrize("bad", ["--output=/etc/passwd", "HEAD..main", "a;rm -rf /", "$(whoami)", ""])
def test_rejects_unsafe_revisions(bad):
    with pytest.raises(ValueError):
        validate_rev(bad)


@pytest.mark.parametrize("bad", ["../secrets.csv", "/etc/passwd", "C:/Windows/x", "-rf", "data/../../x"])
def test_rejects_unsafe_paths(bad):
    with pytest.raises(ValueError):
        normalize_path(bad)


def test_invalid_arguments_return_invalid_request():
    registry = ToolRegistry()
    register_dvc_tools(registry, DVCTools(settings=Settings(), runner=SafeCommandRunner(Settings().repo_root)))
    res = registry.execute("get_dataset_version", {"git_commit": "--exec=evil"})
    assert res.status == ToolStatus.INVALID_REQUEST
    assert "git_commit" in res.error


def test_command_allow_list_blocks_unapproved_commands(tmp_path):
    runner = SafeCommandRunner(tmp_path)
    for binary, args in [
        ("dvc", ["gc", "-w"]),
        ("git", ["push", "origin"]),
        ("git", ["tag", "evil"]),
        ("dvc", ["remote", "remove", "storage"]),
    ]:
        with pytest.raises(Exception, match="not on the approved list"):
            runner.check_allowed(binary, args, allow_mutating=True)
    with pytest.raises(Exception, match="modifies state"):
        runner.check_allowed("dvc", ["checkout", "x.dvc"], allow_mutating=False)


# ----------------------------------------------------- failures (mocked) -----
class _FakeRunner(SafeCommandRunner):
    def __init__(self, exc: BaseException):
        super().__init__(repo_root=Settings().repo_root)
        self.exc = exc

    def run(self, binary, args, **kw):  # noqa: ARG002
        self.check_allowed(binary, args, kw.get("allow_mutating", False))
        raise self.exc


def _registry_with(exc) -> ToolRegistry:
    registry = ToolRegistry()
    register_dvc_tools(registry, DVCTools(settings=Settings(), runner=_FakeRunner(exc)))
    return registry


def test_timeout_is_reported_as_timeout(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    registry = ToolRegistry()
    register_dvc_tools(registry, DVCTools(settings=Settings(repo_root=tmp_path), runner=SafeCommandRunner(tmp_path, 1)))
    res = registry.execute("list_datasets", {})
    assert res.status == ToolStatus.TIMEOUT
    assert res.data is None


def test_unavailable_binary_returns_unavailable_without_fabricated_data():
    from agent.tools.base import ToolUnavailableError

    res = _registry_with(ToolUnavailableError("DVC is not installed")).execute("get_dataset_version", {})
    assert res.status == ToolStatus.UNAVAILABLE
    assert res.data is None  # no invented version/commit


def test_structured_log_has_required_fields_and_no_secrets(caplog):
    from agent.tools.base import ToolError

    caplog.set_level(logging.INFO, logger="mlops_agent")
    leak = "remote https://user:hunter2@example.com/store failed, token=Bearer abc.def.ghi"
    _registry_with(ToolError(leak)).execute("list_datasets", {}, request_id="req-123")
    rec = next(r for r in caplog.records if r.getMessage() == "tool_call")
    f = rec.fields
    assert f["tool"] == "list_datasets" and f["status"] == "error" and f["request_id"] == "req-123"
    assert isinstance(f["duration_ms"], float)
    assert "hunter2" not in str(f) and "abc.def.ghi" not in str(f)


def test_broken_repository_is_unavailable_not_a_false_negative(tmp_path):
    # a directory that is not a git repository must not produce "revision does not exist"
    registry = ToolRegistry()
    settings = Settings(repo_root=tmp_path, default_dataset="data/ds.csv")
    register_dvc_tools(registry, DVCTools(settings=settings, runner=SafeCommandRunner(tmp_path, 30)))
    res = registry.execute("get_dataset_version", {"git_commit": "HEAD"})
    assert res.status == ToolStatus.UNAVAILABLE and res.data is None
