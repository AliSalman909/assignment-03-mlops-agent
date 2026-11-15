"""Task 1 - DVC Data Management module.

Tools exposed to the agent
--------------------------
read-only
  dvc_list_datasets           which datasets/outputs DVC tracks at a revision
  dvc_get_dataset_version     version label + md5 + git commit of a dataset at a revision
                              (used to resolve the dataset behind an MLflow run's git commit)
  dvc_dataset_history         every recorded version of a dataset (git history of its .dvc/.lock)
  dvc_compare_versions        diff two revisions: hashes, size, rows/columns/class balance
  dvc_check_changes           has tracked data drifted from what git records? is it pushed?
  dvc_reproduce_instructions  exact commands to restore a previous dataset version
sandboxed write
  dvc_materialize_version     reproduce a past version into artifacts/reproduced/ via `dvc get`
                              from the configured remote and verify the md5 (workspace untouched)
state-changing (requires explicit confirmation)
  dvc_restore_workspace       `dvc checkout <target>`: discard local drift, restore committed data

Safety model
------------
* Only allow-listed git/dvc sub-commands can run (`SafeCommandRunner.ALLOWED`); anything else
  raises ToolForbiddenError before a process is spawned.
* `shell=False` everywhere; revisions/paths are validated by regex and may not start with "-",
  so user text can never be interpreted as a command option.
* Paths must be relative and stay inside the repository; outputs only go to the sandbox dir.
* Every call has a timeout and stderr is redacted before being surfaced.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from agent.config import Settings, get_settings
from agent.logging_utils import redact
from agent.tools.base import (
    Tool,
    ToolError,
    ToolForbiddenError,
    ToolInvalidRequestError,
    ToolOutput,
    ToolRegistry,
    ToolTimeoutError,
    ToolUnavailableError,
)

MODULE = "dvc"

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
_REV_RE = re.compile(r"^(?!-)[A-Za-z0-9._/~^@{}\-]{1,128}$")


def validate_rev(rev: str) -> str:
    rev = rev.strip()
    if not _REV_RE.match(rev) or ".." in rev:
        raise ValueError(f"'{rev}' is not an acceptable git revision (branch, tag, or commit sha)")
    return rev


def normalize_path(path: str) -> str:
    """Return a clean repo-relative POSIX path for a dataset (without the .dvc suffix)."""
    p = path.strip().replace("\\", "/")
    if p.endswith(".dvc"):
        p = p[: -len(".dvc")]
    if not p or p.startswith("-") or p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        raise ValueError(f"'{path}' must be a relative path inside the repository")
    pure = PurePosixPath(p)
    if any(part in ("..", "") for part in pure.parts):
        raise ValueError(f"'{path}' must not contain '..'")
    if not re.match(r"^[A-Za-z0-9._/\- ]+$", p):
        raise ValueError(f"'{path}' contains unsupported characters")
    return str(pure)


# ---------------------------------------------------------------------------
# Command runner with an allow-list
# ---------------------------------------------------------------------------
class SafeCommandRunner:
    """Runs only approved git/dvc sub-commands. Read-only unless `allow_mutating=True`."""

    # (binary, required leading args) -> mutating?
    ALLOWED: dict[tuple[str, tuple[str, ...]], bool] = {
        ("git", ("rev-parse", "--verify")): False,
        ("git", ("log",)): False,
        ("git", ("show",)): False,
        ("git", ("ls-tree",)): False,
        ("git", ("tag", "--list")): False,
        ("git", ("status", "--porcelain")): False,
        ("git", ("remote", "get-url")): False,
        ("dvc", ("version",)): False,
        ("dvc", ("status",)): False,
        ("dvc", ("diff",)): False,
        ("dvc", ("remote", "list")): False,
        ("dvc", ("remote", "default")): False,
        ("dvc", ("get",)): False,  # writes only to the sandbox dir (checked by the caller)
        ("dvc", ("checkout",)): True,
    }

    def __init__(self, repo_root: Path, timeout_s: float = 60.0) -> None:
        self.repo_root = repo_root
        self.timeout_s = timeout_s

    @staticmethod
    def _dvc_cmd() -> list[str]:
        if importlib.util.find_spec("dvc") is not None:
            return [sys.executable, "-m", "dvc"]
        exe = shutil.which("dvc")
        if exe is None:
            raise ToolUnavailableError("DVC is not installed in the agent environment")
        return [exe]

    def check_allowed(self, binary: str, args: list[str], allow_mutating: bool) -> None:
        for (b, prefix), mutating in self.ALLOWED.items():
            if b == binary and tuple(args[: len(prefix)]) == prefix:
                if mutating and not allow_mutating:
                    raise ToolForbiddenError(f"'{binary} {' '.join(prefix)}' modifies state and is not permitted here")
                return
        raise ToolForbiddenError(f"command '{binary} {' '.join(args[:2])}' is not on the approved list")

    def run(self, binary: str, args: list[str], *, allow_mutating: bool = False, timeout_s: float | None = None) -> str:
        self.check_allowed(binary, args, allow_mutating)
        if binary == "git":
            if shutil.which("git") is None:
                raise ToolUnavailableError("git is not installed")
            cmd = ["git", *args]
        else:
            cmd = [*self._dvc_cmd(), *args]

        env = {**os.environ, "DVC_NO_ANALYTICS": "1", "GIT_TERMINAL_PROMPT": "0", "PYTHONIOENCODING": "utf-8"}
        try:
            proc = subprocess.run(  # noqa: S603 - fixed binary, allow-listed args, shell=False
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s or self.timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolTimeoutError(f"'{binary} {args[0]}' timed out after {exc.timeout:.0f}s") from exc
        except FileNotFoundError as exc:
            raise ToolUnavailableError(f"'{binary}' executable not found") from exc

        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip().splitlines()
            tail = " | ".join(msg[-3:]) if msg else f"exit code {proc.returncode}"
            raise ToolError(redact(f"'{binary} {' '.join(args[:2])}' failed: {tail}"))
        return proc.stdout


# ---------------------------------------------------------------------------
# Tool input models
# ---------------------------------------------------------------------------
class _RevInput(BaseModel):
    rev: str = Field("HEAD", description="Git revision: branch, tag (e.g. 'data-v1.1') or commit sha")

    @field_validator("rev")
    @classmethod
    def _v_rev(cls, v: str) -> str:
        return validate_rev(v)


class _PathInput(BaseModel):
    path: str = Field(
        default_factory=lambda: get_settings().default_dataset,
        description="Repo-relative path of the DVC-tracked dataset, e.g. 'data/raw/churn.csv'",
    )

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: str) -> str:
        return normalize_path(v)


class ListDatasetsInput(_RevInput):
    pass


class GetDatasetVersionInput(_PathInput, _RevInput):
    expected_md5: str | None = Field(
        None,
        description="Optional md5 recorded elsewhere (e.g. an MLflow run tag) to verify against DVC's record",
    )

    @field_validator("expected_md5")
    @classmethod
    def _v_md5(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[0-9a-f]{32}(\.dir)?", v.strip()):
            raise ValueError("expected_md5 must be a 32-char hex md5")
        return v.strip() if v else v


class DatasetHistoryInput(_PathInput):
    max_entries: int = Field(20, ge=1, le=200)


class CompareVersionsInput(_PathInput):
    rev_a: str = Field(..., description="Older revision (tag/commit/branch)")
    rev_b: str = Field("HEAD", description="Newer revision (tag/commit/branch)")
    include_content_stats: bool = Field(True, description="Load both versions and compare rows/columns/labels")

    @field_validator("rev_a", "rev_b")
    @classmethod
    def _v_revs(cls, v: str) -> str:
        return validate_rev(v)


class CheckChangesInput(BaseModel):
    check_remote: bool = Field(True, description="Also check whether cached data is pushed to the DVC remote")


class ReproduceInput(_PathInput, _RevInput):
    pass


class MaterializeInput(_PathInput, _RevInput):
    pass


class RestoreWorkspaceInput(_PathInput):
    pass


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------
class DVCTools:
    def __init__(self, settings: Settings | None = None, runner: SafeCommandRunner | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.repo_root
        self.runner = runner or SafeCommandRunner(self.root, self.settings.dvc_timeout_s)

    # ----- low-level helpers ------------------------------------------------
    def _git(self, *args: str) -> str:
        return self.runner.run("git", list(args))

    def _dvc(self, *args: str, allow_mutating: bool = False) -> str:
        return self.runner.run("dvc", list(args), allow_mutating=allow_mutating)

    def _resolve_commit(self, rev: str) -> str:
        try:
            return self._git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").strip()
        except ToolError as exc:
            raise ToolInvalidRequestError(f"git revision '{rev}' does not exist in this repository") from exc

    def _commit_info(self, sha: str) -> dict[str, str]:
        out = self._git("log", "-1", "--format=%H%x1f%h%x1f%cI%x1f%s", sha).strip()
        full, short, date, subject = out.split("\x1f", 3)
        return {"commit": full, "short_commit": short, "commit_date": date, "commit_message": subject}

    def _show(self, sha: str, repo_path: str) -> str | None:
        try:
            return self._git("show", f"{sha}:{repo_path}")
        except ToolError:
            return None

    def _tracking_record(self, path: str, sha: str) -> dict[str, Any]:
        """Return DVC's recorded hash for `path` at commit `sha` from a .dvc file or dvc.lock."""
        raw = self._show(sha, f"{path}.dvc")
        if raw is not None:
            doc = yaml.safe_load(raw) or {}
            for out in doc.get("outs", []):
                return {
                    "path": path,
                    "tracked_by": f"{path}.dvc",
                    "md5": out.get("md5"),
                    "size_bytes": out.get("size"),
                    "nfiles": out.get("nfiles"),
                    "hash_algo": out.get("hash", "md5"),
                }
        lock = self._show(sha, "dvc.lock")
        if lock is not None:
            doc = yaml.safe_load(lock) or {}
            for stage, body in (doc.get("stages") or {}).items():
                for out in body.get("outs", []):
                    if out.get("path") == path:
                        return {
                            "path": path,
                            "tracked_by": f"dvc.lock (stage '{stage}')",
                            "md5": out.get("md5"),
                            "size_bytes": out.get("size"),
                            "nfiles": out.get("nfiles"),
                            "hash_algo": out.get("hash", "md5"),
                        }
        raise ToolInvalidRequestError(f"'{path}' is not tracked by DVC at commit {sha[:8]}")

    def _version_tags(self) -> dict[str, str]:
        """tag name -> commit sha for dataset version tags (e.g. data-v1.0)."""
        prefix = self.settings.dataset_tag_prefix
        names = [t for t in self._git("tag", "--list", f"{prefix}*").split() if t]
        return {t: self._resolve_commit(t) for t in names}

    def _version_label(self, path: str, md5: str | None, sha: str) -> dict[str, Any]:
        """Identify the dataset version by matching content hash against tagged versions."""
        tags = self._version_tags()
        tags_at_commit = [t for t, c in tags.items() if c == sha]
        matching = []
        for tag, commit in tags.items():
            try:
                if self._tracking_record(path, commit)["md5"] == md5:
                    matching.append(tag)
            except ToolError:
                continue
        prefix = self.settings.dataset_tag_prefix
        label = matching[0][len(prefix):] if matching else None
        return {
            "version": label,
            "version_tag": matching[0] if matching else None,
            "identified_by": "content md5 matches tagged version" if matching else "no tagged version has this md5",
            "tags_at_commit": tags_at_commit,
        }

    def _remote_info(self) -> dict[str, Any]:
        try:
            listing = self._dvc("remote", "list").strip().splitlines()
        except ToolError:
            listing = []
        remotes = {}
        for line in listing:
            parts = line.split()
            if len(parts) >= 2:
                remotes[parts[0]] = redact(parts[1])
        try:
            default = self._dvc("remote", "default").strip() or None
        except ToolError:
            default = None
        return {"default_remote": default, "remotes": remotes}

    def _cache_path(self, md5: str) -> Path:
        return self.root / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]

    def _remote_object_path(self, md5: str) -> Path | None:
        info = self._remote_info()
        url = info["remotes"].get(info["default_remote"] or "", "")
        if not url or "://" in url:
            return None  # cloud remote - existence checked via dvc status --cloud instead
        base = Path(url)
        if not base.is_absolute():
            base = (self.root / ".dvc" / base).resolve()
        return base / "files" / "md5" / md5[:2] / md5[2:]

    def _load_frame(self, path: str, rev: str):
        """Read a version's content via the DVC Python API (cache first, then remote)."""
        import dvc.api
        import pandas as pd

        try:
            text = dvc.api.read(path, repo=str(self.root), rev=rev, mode="r", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not load {path}@{rev} from DVC cache/remote: {type(exc).__name__}") from exc
        return pd.read_csv(io.StringIO(text))

    @staticmethod
    def _frame_stats(df) -> dict[str, Any]:
        stats: dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns), "duplicate_rows": int(df.duplicated().sum())}
        if "churned" in df.columns:
            stats["positive_rate"] = round(float(df["churned"].mean()), 4)
        return stats

    @staticmethod
    def _md5_file(path: Path) -> str:
        h = hashlib.md5()  # noqa: S324 - DVC content addressing, not security
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ----- tools ------------------------------------------------------------
    def list_datasets(self, p: ListDatasetsInput) -> ToolOutput:
        sha = self._resolve_commit(p.rev)
        files = self._git("ls-tree", "-r", "--name-only", sha).splitlines()
        datasets = []
        for f in files:
            if f.endswith(".dvc") and not f.startswith(".dvc/"):
                datasets.append(self._tracking_record(f[: -len(".dvc")], sha) | {"kind": "dvc add (raw input)"})
        if "dvc.lock" in files:
            doc = yaml.safe_load(self._show(sha, "dvc.lock") or "") or {}
            for stage, body in (doc.get("stages") or {}).items():
                for out in body.get("outs", []):
                    datasets.append(
                        {
                            "path": out["path"],
                            "tracked_by": f"dvc.lock (stage '{stage}')",
                            "md5": out.get("md5"),
                            "size_bytes": out.get("size"),
                            "kind": "pipeline output",
                        }
                    )
        return ToolOutput(
            data={"revision": p.rev, "commit": sha, "count": len(datasets), "datasets": datasets},
            evidence_source=[f"git ls-tree {sha[:8]}", "*.dvc files", "dvc.lock"],
        )

    def get_dataset_version(self, p: GetDatasetVersionInput) -> ToolOutput:
        sha = self._resolve_commit(p.rev)
        record = self._tracking_record(p.path, sha)
        # the commit that last changed this dataset (may be older than `rev`)
        tracker = record["tracked_by"].split(" ")[0]
        last_change = self._git("log", "-1", "--format=%H", sha, "--", tracker).strip() or sha
        data: dict[str, Any] = {
            "requested_revision": p.rev,
            **self._commit_info(sha),
            **record,
            **self._version_label(p.path, record["md5"], sha),
            "data_last_changed_in_commit": last_change,
            **self._remote_info(),
            "in_local_cache": self._cache_path(record["md5"]).exists() if record["md5"] else False,
        }
        if p.expected_md5:
            data["expected_md5"] = p.expected_md5
            data["md5_matches_expected"] = p.expected_md5 == record["md5"]
        return ToolOutput(data=data, evidence_source=[f"git show {sha[:8]}:{tracker}", "git tags", "dvc remote list"])

    def dataset_history(self, p: DatasetHistoryInput) -> ToolOutput:
        head = self._resolve_commit("HEAD")
        record = self._tracking_record(p.path, head)
        tracker = record["tracked_by"].split(" ")[0]
        log = self._git("log", f"-n{p.max_entries}", "--format=%H%x1f%h%x1f%cI%x1f%s", "HEAD", "--", tracker)
        tags = self._version_tags()
        by_commit: dict[str, list[str]] = {}
        for t, c in tags.items():
            by_commit.setdefault(c, []).append(t)
        versions, seen = [], set()
        for line in log.strip().splitlines():
            full, short, date, subject = line.split("\x1f", 3)
            try:
                rec = self._tracking_record(p.path, full)
            except ToolError:
                continue
            if rec["md5"] in seen:
                continue
            seen.add(rec["md5"])
            versions.append(
                {
                    "commit": full,
                    "short_commit": short,
                    "commit_date": date,
                    "commit_message": subject,
                    "md5": rec["md5"],
                    "size_bytes": rec["size_bytes"],
                    "tags": by_commit.get(full, []),
                }
            )
        return ToolOutput(
            data={"path": p.path, "tracked_by": tracker, "versions": versions, "count": len(versions)},
            evidence_source=[f"git log -- {tracker}", "git tags"],
        )

    def compare_versions(self, p: CompareVersionsInput) -> ToolOutput:
        sha_a, sha_b = self._resolve_commit(p.rev_a), self._resolve_commit(p.rev_b)
        rec_a, rec_b = self._tracking_record(p.path, sha_a), self._tracking_record(p.path, sha_b)
        data: dict[str, Any] = {
            "path": p.path,
            "a": {"revision": p.rev_a, "commit": sha_a, "md5": rec_a["md5"], "size_bytes": rec_a["size_bytes"],
                  **self._version_label(p.path, rec_a["md5"], sha_a)},
            "b": {"revision": p.rev_b, "commit": sha_b, "md5": rec_b["md5"], "size_bytes": rec_b["size_bytes"],
                  **self._version_label(p.path, rec_b["md5"], sha_b)},
            "content_identical": rec_a["md5"] == rec_b["md5"],
        }
        if rec_a["size_bytes"] is not None and rec_b["size_bytes"] is not None:
            data["size_delta_bytes"] = rec_b["size_bytes"] - rec_a["size_bytes"]

        sources = [f"git show {sha_a[:8]}/{sha_b[:8]} DVC records", f"dvc diff {sha_a[:8]} {sha_b[:8]}"]
        import json

        try:
            diff = json.loads(self._dvc("diff", "--json", sha_a, sha_b) or "{}")
            data["dvc_diff"] = {k: [e.get("path") for e in v] for k, v in diff.items() if v}
        except (ToolError, ValueError) as exc:
            data["dvc_diff_error"] = str(exc)

        if p.include_content_stats and not data["content_identical"]:
            try:
                sa, sb = self._frame_stats(self._load_frame(p.path, sha_a)), self._frame_stats(self._load_frame(p.path, sha_b))
                data["content"] = {
                    "a": sa,
                    "b": sb,
                    "row_delta": sb["rows"] - sa["rows"],
                    "columns_added": [c for c in sb["columns"] if c not in sa["columns"]],
                    "columns_removed": [c for c in sa["columns"] if c not in sb["columns"]],
                }
                sources.append("dvc.api.read (content statistics)")
            except ToolError as exc:
                data["content_error"] = str(exc)
        return ToolOutput(data=data, evidence_source=sources)

    def check_changes(self, p: CheckChangesInput) -> ToolOutput:
        import json

        status = json.loads(self._dvc("status", "--json") or "{}")
        workspace_changes = []
        stale_stages = []
        for target, entries in status.items():
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for kind, detail in entry.items():
                    item = {"target": target, "change": kind, "detail": detail}
                    (workspace_changes if target.endswith(".dvc") else stale_stages).append(item)

        porcelain = self._git("status", "--porcelain", "--", "*.dvc", "dvc.lock", "dvc.yaml").strip()
        uncommitted = [ln[3:] for ln in porcelain.splitlines() if ln]

        data: dict[str, Any] = {
            "data_changed": bool(workspace_changes),
            "workspace_changes": workspace_changes,
            "pipeline_stale": bool(stale_stages),
            "stale_stages": stale_stages,
            "uncommitted_dvc_metadata": uncommitted,
        }
        sources = ["dvc status --json", "git status --porcelain"]
        if p.check_remote:
            try:
                cloud = json.loads(self._dvc("status", "--cloud", "--json") or "{}")
                data["not_pushed_to_remote"] = [k for k, v in cloud.items() if v in ("new", "modified")]
                data["missing_locally"] = [k for k, v in cloud.items() if v == "deleted"]
                sources.append("dvc status --cloud --json")
            except ToolError as exc:
                data["remote_check_error"] = str(exc)
        data["summary"] = (
            "tracked data differs from the committed DVC version"
            if workspace_changes
            else "workspace data matches the committed DVC version"
        )
        return ToolOutput(data=data, evidence_source=sources)

    def reproduce_instructions(self, p: ReproduceInput) -> ToolOutput:
        sha = self._resolve_commit(p.rev)
        record = self._tracking_record(p.path, sha)
        version = self._version_label(p.path, record["md5"], sha)
        tracker = record["tracked_by"].split(" ")[0]
        remote = self._remote_info()
        remote_obj = self._remote_object_path(record["md5"]) if record["md5"] else None
        short = sha[:8]
        steps = [
            {"step": 1, "purpose": "restore the DVC pointer recorded at that commit",
             "command": f"git checkout {short} -- {tracker}"},
            {"step": 2, "purpose": "fetch the exact bytes from the remote (or local cache) and place them in the workspace",
             "command": f"dvc pull {tracker}" if tracker.endswith(".dvc") else f"dvc pull {p.path}"},
            {"step": 3, "purpose": "verify the workspace matches the recorded hash (expects no changes)",
             "command": f"dvc status {tracker}" if tracker.endswith(".dvc") else "dvc status"},
            {"step": 4, "purpose": "return to the current version afterwards",
             "command": f"git checkout HEAD -- {tracker} && dvc checkout {tracker if tracker.endswith('.dvc') else ''}".strip()},
        ]
        alternative = {
            "purpose": "non-destructive copy without touching the workspace",
            "command": f"dvc get . {p.path} --rev {short} -o artifacts/reproduced/{short}/{PurePosixPath(p.path).name}",
            "agent_tool": "dvc_materialize_version",
        }
        return ToolOutput(
            data={
                "path": p.path,
                "revision": p.rev,
                "commit": sha,
                **version,
                "expected_md5": record["md5"],
                "expected_size_bytes": record["size_bytes"],
                "steps": steps,
                "non_destructive_alternative": alternative,
                "downstream": "run `dvc repro` afterwards to rebuild pipeline outputs (data/processed) for this version",
                "availability": {
                    "in_local_cache": self._cache_path(record["md5"]).exists() if record["md5"] else False,
                    "in_default_remote": remote_obj.exists() if remote_obj else "unknown (cloud remote)",
                    **remote,
                },
            },
            evidence_source=[f"git show {short}:{tracker}", "dvc remote list", "local cache/remote object lookup"],
        )

    def materialize_version(self, p: MaterializeInput) -> ToolOutput:
        sha = self._resolve_commit(p.rev)
        record = self._tracking_record(p.path, sha)
        if record.get("nfiles"):
            raise ToolInvalidRequestError("materializing directory datasets is not supported by this tool")
        sandbox = (self.root / self.settings.reproduce_dir).resolve()
        out_dir = (sandbox / sha[:8]).resolve()
        if sandbox not in out_dir.parents:
            raise ToolForbiddenError("output path escapes the reproduction sandbox")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / PurePosixPath(p.path).name
        if out_file.exists():
            out_file.unlink()

        # `dvc get` clones the repo at that commit and fetches the object from the configured remote,
        # which proves the version is reproducible from shared storage - not just from this workspace.
        self.runner.run("dvc", ["get", str(self.root), p.path, "--rev", sha, "-o", str(out_file)], timeout_s=max(
            self.settings.dvc_timeout_s, 120))
        actual = self._md5_file(out_file)
        verified = actual == record["md5"]
        if not verified:
            raise ToolError(f"reproduced file md5 {actual} does not match recorded {record['md5']}")
        return ToolOutput(
            data={
                "path": p.path,
                "commit": sha,
                **self._version_label(p.path, record["md5"], sha),
                "output_file": str(out_file.relative_to(self.root)).replace("\\", "/"),
                "recorded_md5": record["md5"],
                "reproduced_md5": actual,
                "verified": verified,
                "size_bytes": out_file.stat().st_size,
            },
            evidence_source=[f"dvc get --rev {sha[:8]}", "md5 verification of reproduced bytes"],
        )

    def restore_workspace(self, p: RestoreWorkspaceInput) -> ToolOutput:
        record = self._tracking_record(p.path, self._resolve_commit("HEAD"))
        tracker = record["tracked_by"].split(" ")[0]
        target = tracker if tracker.endswith(".dvc") else p.path
        self._dvc("checkout", "--force", target, allow_mutating=True)
        actual = self._md5_file(self.root / p.path) if (self.root / p.path).is_file() else None
        return ToolOutput(
            data={"path": p.path, "restored_to_md5": record["md5"], "workspace_md5": actual,
                  "verified": actual == record["md5"]},
            evidence_source=[f"dvc checkout {target}", "md5 verification"],
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_dvc_tools(registry: ToolRegistry, tools: DVCTools | None = None) -> DVCTools:
    t = tools or DVCTools()
    specs = [
        ("dvc_list_datasets", ListDatasetsInput, t.list_datasets, False, False,
         "List every dataset tracked by DVC at a git revision (raw inputs from .dvc files and pipeline "
         "outputs from dvc.lock) with their md5 and size."),
        ("dvc_get_dataset_version", GetDatasetVersionInput, t.get_dataset_version, False, False,
         "Get the dataset version label, md5, size, and git commit for a dataset at a revision. Pass the git "
         "commit recorded on an MLflow run as `rev` to find which dataset version trained it; pass the run's "
         "recorded md5 as `expected_md5` to verify provenance."),
        ("dvc_dataset_history", DatasetHistoryInput, t.dataset_history, False, False,
         "List all historical versions of a dataset (each distinct md5 with its commit, date, and version tags)."),
        ("dvc_compare_versions", CompareVersionsInput, t.compare_versions, False, False,
         "Compare a dataset between two revisions: hash/size change, dvc diff, and row/column/label-balance changes."),
        ("dvc_check_changes", CheckChangesInput, t.check_changes, False, False,
         "Detect whether tracked data in the workspace differs from the committed DVC version, whether the "
         "pipeline is stale, and whether data is pushed to the DVC remote."),
        ("dvc_reproduce_instructions", ReproduceInput, t.reproduce_instructions, False, False,
         "Explain the exact commands needed to reproduce a previous dataset version, with expected md5 and "
         "whether the bytes are available in the cache/remote. Does not execute anything."),
        ("dvc_materialize_version", MaterializeInput, t.materialize_version, False, False,
         "Reproduce a past dataset version into the sandbox directory artifacts/reproduced/<commit>/ using "
         "`dvc get` from the remote, and verify its md5 against the record. Does not modify the workspace."),
        ("dvc_restore_workspace", RestoreWorkspaceInput, t.restore_workspace, True, True,
         "STATE-CHANGING: run `dvc checkout` to discard local modifications and restore the committed version "
         "of a dataset in the workspace. Requires explicit user confirmation."),
    ]
    for name, model, func, mutating, confirm, desc in specs:
        registry.register(Tool(name=name, module=MODULE, description=desc, input_model=model, func=func,
                               mutating=mutating, requires_confirmation=confirm))
    return t
