# MLOps AI Agent

A conversational agent that answers operational questions about an ML system using **real tool
evidence** from DVC (data versions), MLflow (experiments), Airflow (training pipeline), and
Kubernetes (serving), and never from model memory.

| Module | Status |
|---|---|
| Task 1: DVC data management | ✅ implemented + tested |
| Task 2: MLflow experiment intelligence | ⏳ |
| Task 3: Airflow workflow agent | ⏳ |
| Task 4: Kubernetes operations | ⏳ |
| Integration: LLM orchestrator + FastAPI | ⏳ |

## Repository layout

```
agent/
  config.py            settings from environment variables
  logging_utils.py     JSON logs with secret redaction
  schemas.py           ToolResult envelope / ToolSpec (provider-agnostic tool definitions)
  cli.py               call any tool directly: python -m agent.cli <tool> '<json>'
  tools/
    base.py            registry + execute_tool (validation, confirmation policy, error mapping, logging)
    dvc_tools.py       Task 1
scripts/generate_dataset.py   deterministic dataset generator (v1.0, v1.1, v2.0)
training/preprocess.py        DVC pipeline stage (clean + train/test split + stats.json)
dvc.yaml / dvc.lock           pipeline definition and recorded hashes
data/raw/churn.csv.dvc        DVC pointer to the raw dataset
docs/evidence/dvc/            exported tool outputs used in the report
tests/                        pytest suite (runs against a throw-away git+DVC repo)
```

## Setup (Windows / Linux / macOS)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -e ".[dev]"
dvc pull            # fetch data from the configured remote
dvc status          # -> "Data and pipelines are up to date."
```

> **Windows note:** always activate the venv before running `dvc repro`. The pipeline stage calls
> `python`, and outside the venv that can resolve to the Microsoft Store alias, which hangs.

### DVC remote

The demo uses a local remote at `.dvcstore/` (git-ignored). In production you would point it at
object storage, e.g. `dvc remote add -d storage s3://bucket/path`, with credentials supplied
through environment variables or `dvc remote modify --local` (never committed).

## Dataset versions

| Tag | Rows | Change |
|---|---|---|
| `data-v1.0` | 4,040 | initial drop (contains 40 duplicates) |
| `data-v1.1` | 6,060 | +2k newly collected customers |
| `data-v2.0` | 6,000 | new `last_login_days` feature, deduplicated, less label noise |

Each tag is an annotated git tag on the commit that changed `data/raw/churn.csv.dvc`. Every
version's bytes are pushed to the remote, so any of them can be restored exactly.

## Task 1: DVC tools

| Tool | Kind | What it does |
|---|---|---|
| `dvc_list_datasets` | read | datasets tracked at a revision (`.dvc` files + `dvc.lock` outputs) |
| `dvc_get_dataset_version` | read | version label, md5, size, commit for a dataset at a revision; optional `expected_md5` provenance check |
| `dvc_dataset_history` | read | every distinct version of a dataset with commits and tags |
| `dvc_compare_versions` | read | hash/size diff, `dvc diff`, row/column/label-balance changes |
| `dvc_check_changes` | read | workspace drift vs. committed version, stale pipeline stages, unpushed data |
| `dvc_reproduce_instructions` | read | exact commands to restore a past version + cache/remote availability |
| `dvc_materialize_version` | sandbox write | `dvc get` a past version into `artifacts/reproduced/<commit>/` and verify its md5 |
| `dvc_restore_workspace` | **state-changing** | `dvc checkout` to discard local drift; **requires explicit confirmation** |

**How a version is identified.** A dataset's version is not taken from the commit it is
requested at. The tool reads the md5 recorded in the `.dvc` file (or `dvc.lock`) at that commit
and matches it against the md5 recorded at every `data-v*` tag. A code-only commit made after
`data-v2.0` is therefore still reported as `v2.0`, along with the commit where the data last
changed. When MLflow runs log their git commit (Task 2), this is how the agent answers *"which
dataset version trained experiment N?"*.

**Safety controls**
- Only allow-listed `git`/`dvc` sub-commands can run; anything else is refused before a process starts.
- `shell=False`; revisions and paths are regex-validated and may not start with `-` (no option
  injection), contain `..`, or be absolute.
- Every call has a timeout (`DVC_TIMEOUT_S`, default 60s).
- Errors are typed (`ok`, `error`, `unavailable`, `timeout`, `invalid_request`,
  `confirmation_required`, `forbidden`). A failed tool returns `data: null`, so nothing can be
  mistaken for evidence.
- Each call logs one JSON line with tool, status, duration and request id; credentials in
  messages and URLs are redacted.

### Try it

```bash
python -m agent.cli list
python -m agent.cli dvc_get_dataset_version '{"rev": "data-v1.1"}'
python -m agent.cli dvc_compare_versions '{"rev_a": "data-v1.0", "rev_b": "data-v2.0"}'
python -m agent.cli dvc_materialize_version '{"rev": "data-v1.0"}'
python -m agent.cli dvc_check_changes
python -m agent.cli dvc_restore_workspace '{}' --confirm
```

(On PowerShell, wrap the JSON in single quotes and escape inner quotes as `\"`.)

## Tests

```bash
pytest -q        # 24 tests; builds a temporary git+DVC repo, external failures are mocked
ruff check .
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MLOPS_REPO_ROOT` | project root | repository the DVC tools inspect |
| `DVC_TIMEOUT_S` | `60` | per-command timeout |
| `DVC_REPRODUCE_DIR` | `artifacts/reproduced` | sandbox for materialized versions |
| `DVC_DEFAULT_DATASET` | `data/raw/churn.csv` | dataset used when none is specified |
| `DVC_DATASET_TAG_PREFIX` | `data-` | prefix of dataset version tags |
