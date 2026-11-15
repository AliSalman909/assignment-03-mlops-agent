# MLOps AI Agent

A conversational agent that answers operational questions about an ML system using **real tool
evidence** from DVC (data versions), MLflow (experiments), Airflow (training pipeline), and
Kubernetes (serving), and never from model memory.

| Module | Status |
|---|---|
| Task 1: DVC data management | ✅ implemented + tested |
| Task 2: MLflow experiment intelligence | ✅ implemented + tested |
| Task 3: Airflow workflow agent + training DAG | ✅ implemented + tested |
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
    mlflow_tools.py    Task 2 (MlflowClient, read-only)
    airflow_tools.py   Task 3 (Airflow 3 REST API v2, JWT auth)
dags/training_pipeline.py     Airflow training DAG
docker/airflow/               Airflow 3.4 image (+ scikit-learn, mlflow-skinny, skops)
docker-compose.yml            MLflow 3.18 server + Airflow 3.4
scripts/generate_dataset.py   deterministic dataset generator (v1.0, v1.1, v2.0)
scripts/seed_experiments.py   provenance-tagged MLflow history + 'champion' model
training/preprocess.py        DVC pipeline stage (clean + train/test split + stats.json)
training/train.py             shared validation / training / MLflow logging / quality gate
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

All parameters are optional unless marked *required*. `dataset_path` defaults to `data/raw/churn.csv`;
`git_commit` defaults to `HEAD`. Revision parameters accept a commit sha, a version label (`v1.1`),
a tag (`data-v1.1`), or a branch.

| Tool | Parameters | Kind | What it does |
|---|---|---|---|
| `list_datasets` | `git_commit` | read | datasets tracked at a revision (`.dvc` files + `dvc.lock` outputs) |
| `get_dataset_version` | `dataset_path`, `git_commit`, `expected_md5` | read | version label, md5, size and git commit; optional provenance check against an md5 recorded in MLflow |
| `get_dataset_history` | `dataset_path`, `max_entries` | read | every distinct version of a dataset with commits and tags |
| `compare_dataset_versions` | `version_a` *(required)*, `version_b`, `dataset_path` | read | hash/size diff, `dvc diff`, row/column/label-balance changes |
| `check_dataset_changes` | `check_remote` | read | workspace drift vs. committed version, stale pipeline stages, unpushed data |
| `explain_dataset_reproduction` | `git_commit`, `dataset_path` | read | exact commands to restore a past version + cache/remote availability |
| `reproduce_dataset` | `git_commit`, `dataset_path` | sandbox write | `dvc get` a past version from the remote into `artifacts/reproduced/<commit>/` and verify its md5 (workspace untouched) |
| `restore_dataset_workspace` | `dataset_path` | **state-changing** | `dvc checkout` to discard local drift; **requires explicit confirmation** |

Regenerate the report evidence (includes the injected tampering failure, always restores data):
`bash scripts/collect_dvc_evidence.sh`

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
python -m agent.cli get_dataset_version '{"git_commit": "v1.1"}'
python -m agent.cli compare_dataset_versions '{"version_a": "v1.0", "version_b": "v2.0"}'
python -m agent.cli reproduce_dataset '{"git_commit": "data-v1.0"}'
python -m agent.cli check_dataset_changes
python -m agent.cli restore_dataset_workspace '{}' --confirm
```

(On PowerShell, wrap the JSON in single quotes and escape inner quotes as `\"`.)

## Tests

```bash
pytest -q        # 44 tests; DVC uses a temp git+DVC repo, MLflow/Airflow are mocked
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
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `AIRFLOW_BASE_URL` | `http://localhost:8080` | Airflow API |
| `AIRFLOW_USERNAME` / `AIRFLOW_ADMIN_PASSWORD` | `admin` / - | Airflow credentials (also used by docker compose) |
| `AIRFLOW_APPROVED_DAGS` | `churn_training_pipeline` | DAGs the agent may trigger |
| `GEMINI_API_KEY` | - | Gemini API key (put it in `.env`, which is git-ignored) |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | model used by the agent |

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`.

## Running the services (MLflow + Airflow)

```bash
cp .env.example .env            # set AIRFLOW_ADMIN_PASSWORD (and GEMINI_API_KEY)
docker compose up -d --build    # MLflow http://localhost:5000, Airflow http://localhost:8080 (user admin)
python scripts/seed_experiments.py   # one-off: 9 provenance-tagged runs + champion model
```

The project folder is mounted **read-only** into Airflow, so the pipeline can't modify the repo.

## Task 2: MLflow tools (read-only, `MlflowClient`)

`experiment_id` accepts an id (`1`) or a name (`churn-baseline`).

| Tool | Parameters | What it does |
|---|---|---|
| `list_experiments` | `include_deleted` | experiments with run counts |
| `list_training_runs` | `experiment_id`, `limit`, `filter_string` | recent runs: metrics, model type, provenance tags |
| `get_run_details` | `run_id` *(required)* | metrics, params, tags, artifacts, logged models, registered versions |
| `find_best_model` | `experiment_id`, `metric`, `order` (`max`/`DESC` or `min`/`ASC`) | best run + production-readiness assessment |
| `compare_model_runs` | `run_ids` (or `last_n` + `experiment_id`), `metric` | side-by-side comparison, best per metric, reasoned recommendation |
| `compare_experiments` | `experiment_ids` *(≥2)*, `metric` | best/mean/std per experiment, model types |
| `get_model_versions` | `model_name` | versions, aliases (`champion` = serving, `candidate` = awaiting review), source-run metrics and provenance |

**The highest metric is never treated as "production-ready".** `find_best_model` and `compare_model_runs`
report the margin over the runner-up (and flag it when it's within noise), trade-offs on other
metrics, the quality-gate result, whether dataset provenance was recorded, and registry status,
followed by the checks that still need a human.

**Provenance contract.** Every run is tagged with `git_commit` and `dataset_md5`. Passing them to
`get_dataset_version(git_commit=..., expected_md5=...)` returns the exact dataset version
(e.g. *candidate v4 → data v2.0, champion v1 → data v1.1*, both with verified md5 matches).

## Task 3: Airflow training pipeline and tools

`dags/training_pipeline.py` (`churn_training_pipeline`):

```
validate_dataset → preprocess_data → train_model → evaluate_and_log (MLflow) → quality_gate → register_candidate
```

- **Retries:** 2 per task with a 10s delay. Deterministic failures (bad data, failed gate) raise
  `AirflowFailException` and fail immediately without retrying.
- **Promotion safety:** `register_candidate` runs only if every upstream task succeeded. Even then
  the model gets the `candidate` alias; promotion to `champion` is a human decision.
- **Quality gate:** `f1 ≥ 0.48`, `roc_auc ≥ 0.70`, and no more than 0.02 F1 regression vs. the champion
  (thresholds calibrated on the seeded runs; overridable per run).
- **Logging:** each validation check, dataset hash, metrics and gate decision is written to the task log.
- **Failure injection** via trigger `config`: `{"inject_failure": "corrupt_data"}`,
  `{"inject_failure": "transient_error"}`, or `{"min_f1": 0.60}` (forces a gate failure).

| Tool | Parameters | Kind | What it does |
|---|---|---|---|
| `list_dags` | `limit` | read | DAGs, paused state, tags, whether the agent may trigger them |
| `get_dag_runs` | `dag_id`, `limit` | read | recent runs with state, timing, conf |
| `get_dag_run_status` | `dag_id`, `run_id`, `wait_seconds` | read | per-task states and attempts; can wait for completion |
| `trigger_training_pipeline` | `dag_id`, `config`, `wait_seconds` | **state-changing** | **requires confirmation**, DAG must be on `AIRFLOW_APPROVED_DAGS`, config keys allow-listed |
| `get_failed_tasks` | `dag_id`, `run_id` | read | failed tasks and the downstream tasks they blocked |
| `get_task_logs` | `dag_id`, `run_id`, `task_id`, `try_number` | read | log tail of an attempt (secrets redacted) |
| `diagnose_dag_run` | `dag_id`, `run_id` | read | root task, error excerpt, failure layer, retries, promotion blocked?, next step |

Regenerate MLflow/Airflow evidence (runs the 3 injected failures; needs the stack up):
`bash scripts/collect_mlflow_airflow_evidence.sh`

### Injected-failure results (from `docs/evidence/airflow/`)

| Injection | Agent diagnosis | Promotion |
|---|---|---|
| `corrupt_data` | `validate_dataset` failed, **data layer**: `missing=['churned']`, duplicates 11.8% | blocked (5 tasks `upstream_failed`) |
| `min_f1: 0.60` | `quality_gate` failed, **model layer**: `f1 0.4970 < min_f1 0.6` | blocked (`register_candidate` never ran) |
| `transient_error` | `preprocess_data` failed on attempt 1, succeeded on retry: **recovered** | allowed |
