# MLOps AI Agent

A conversational agent that answers operational questions about an ML system using **real tool
evidence** from DVC (data versions), MLflow (experiments), Airflow (training pipeline), and
Kubernetes (serving), and never from model memory.

| Module | Status |
|---|---|
| Task 1: DVC data management | ✅ implemented + tested |
| Task 2: MLflow experiment intelligence | ✅ implemented + tested |
| Task 3: Airflow workflow agent + training DAG | ✅ implemented + tested |
| Task 4: Kubernetes operations + FastAPI inference service | ✅ implemented + tested |
| Task 5: Multi-tool agent (Gemini) + FastAPI interface | ✅ implemented + tested |

See [architecture.md](architecture.md) for the design, and [docs/conversations/](docs/conversations/) for
the five assessed scenarios run end-to-end with real tool evidence.

## Quick start: talk to the agent

```bash
# prerequisites: venv installed, .env filled in (GEMINI_API_KEY, AIRFLOW_ADMIN_PASSWORD), services running:
docker compose up -d                                  # MLflow + Airflow
kind create cluster --config kubernetes/kind-cluster.yaml && bash scripts/deploy_serving.sh
bash scripts/k8s_agent_kubeconfig.sh                  # restricted agent credentials

python -m agent.main                                  # interactive chat (asks before any state change)
uvicorn api.main:app --port 8100                      # HTTP API, docs at http://localhost:8100/docs
```

```bash
curl -s localhost:8100/agent/query -H 'Content-Type: application/json' \
  -d '{"message": "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}'
# state-changing requests come back with pending_actions; run one only by confirming its id:
curl -s localhost:8100/agent/confirm -H 'Content-Type: application/json' -d '{"action_id": "<id>"}'
```

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
    kubernetes_tools.py Task 4 (official Python client, namespace allow-list, RBAC)
dags/training_pipeline.py     Airflow training DAG
serving/                      FastAPI inference service (app.py, Dockerfile, export_model.py)
kubernetes/                   namespace, configmap, deployment, service, agent-rbac, kind-cluster
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

Before committing/pushing: `bash scripts/check_secrets.sh` (scans staged changes and tracked files for the values in `.env`/`.secrets/` and for credential patterns; never prints them).

```bash
pytest -q        # 71 tests; DVC uses a temp git+DVC repo; MLflow, Airflow and Kubernetes are mocked
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
| `K8S_KUBECONFIG` | (default kubeconfig) | kubeconfig the agent uses; use the restricted one in `.secrets/` |
| `K8S_NAMESPACE` / `K8S_APPROVED_NAMESPACES` | `mlops` / `mlops` | default namespace / namespaces the agent may touch |
| `K8S_MANAGED_LABEL` | `app.kubernetes.io/part-of=mlops-agent` | only Deployments with this label can be modified |
| `GEMINI_API_KEY` | - | Gemini API key (put it in `.env`, which is git-ignored) |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | model used by the agent (any Gemini model id; `*-latest` aliases resolve server-side, the concrete version is reported in each response's `model` field) |
| `GEMINI_API_VERSION` | `v1beta` | Gemini API version (`*-latest` aliases exist only on `v1beta`) |
| `GEMINI_TEMPERATURE` | `0.1` | sampling temperature (low = consistent tool use) |
| `AGENT_API_KEY` | (unset = open) | require `X-API-Key` on the agent API |
| `AGENT_MAX_STEPS` / `AGENT_MAX_TOOL_CALLS` | `10` / `25` | agent loop budgets |
| `RETRAIN_MAX_MODEL_AGE_DAYS` / `RETRAIN_MIN_PRODUCTION_F1` | `30` / `0.45` | retraining policy thresholds |

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

## Task 4: Kubernetes deployment and operations

### The application

`serving/app.py` is a FastAPI inference service for the registered **champion** model.
`serving/export_model.py` pulls it from the MLflow registry into the image together with its
lineage (registry version, run id, dataset version verified through the DVC tool).

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | liveness: the process is responsive |
| `GET /health/ready` | readiness: the model is loaded (503 otherwise, so Kubernetes stops routing to it) |
| `GET /version` | app version + model lineage |
| `POST /predict` | churn probability + label for 1-1000 customers (validated input) |

Manifests (`kubernetes/`): **Deployment** with 2 replicas, rolling update `maxUnavailable: 0`,
**startup/readiness/liveness probes**, **CPU/memory requests and limits**, non-root with all
capabilities dropped, and model lineage annotations. Plus a **ClusterIP Service** and a
**ConfigMap** for non-sensitive settings.

```bash
kind create cluster --config kubernetes/kind-cluster.yaml   # local cluster in Docker (or use Docker Desktop k8s)
bash scripts/deploy_serving.sh                              # export model -> build -> kind load -> apply
bash scripts/k8s_agent_kubeconfig.sh                        # restricted agent credentials -> .secrets/
```

### Agent permissions (defence in depth)

1. **Tool layer:** every namespace is checked against `K8S_APPROVED_NAMESPACES`. Restart, scale
   and rollback require **explicit confirmation**, only touch Deployments labelled
   `app.kubernetes.io/part-of=mlops-agent`, and scaling is capped at `K8S_MAX_REPLICAS`.
2. **Credentials:** `kubernetes/agent-rbac.yaml` gives the agent's ServiceAccount read access to
   the app's resources and `patch` on Deployments, **in `mlops` only**. Verified with `kubectl auth can-i`:
   it can't delete pods, read secrets, or see `kube-system`.

### Tools

| Tool | Parameters | Kind | What it does |
|---|---|---|---|
| `list_pods` | `namespace`, `label_selector` | read | pods, phase, readiness, restarts, detected problems |
| `list_deployments` | `namespace` | read | desired/ready/updated replicas, images, revision, rollout state |
| `get_pod_status` | `pod_name`, `namespace` | read | container states, last termination (OOMKilled...), probes, resources, events |
| `get_pod_logs` | `pod_name`, `namespace`, `tail_lines`, `previous` | read | container logs (or the crashed instance's), redacted |
| `list_services` | `namespace` | read | type, ports, selector, ready endpoints |
| `inspect_service` | `service_name`, `namespace`, `check_health` | read | endpoints, matching pods, live `/health/ready` through the API server |
| `find_unhealthy_pods` | `namespace` | read | not-ready / CrashLoopBackOff / ImagePullBackOff / OOMKilled pods, restart and probe events |
| `get_events` | `namespace`, `object_name`, `warnings_only`, `limit` | read | events, newest first |
| `get_deployed_version` | `deployment_name`, `namespace` | read | image, app version, model lineage, revision, live `/version` |
| `diagnose_service` | `service_name`, `namespace` | read | "why is my prediction API unavailable?": evidence-based diagnosis + action |
| `restart_deployment` | `deployment_name`, `namespace` | **state-changing** | rolling restart (**confirmation required**) |
| `scale_deployment` | `deployment_name`, `replicas`, `namespace` | **state-changing** | bounded scaling (**confirmation required**) |
| `rollback_deployment` | `deployment_name`, `to_revision`, `namespace` | **state-changing** | roll back to a previous ReplicaSet revision (**confirmation required**) |

### Injected outages (`bash scripts/collect_k8s_evidence.sh`, evidence in `docs/evidence/kubernetes/`)

| Injection | Agent diagnosis | Resolution |
|---|---|---|
| ConfigMap `MODEL_DIR` set to a missing path, pods recycled | **UNAVAILABLE, 0 ready endpoints**. Readiness probe 503 + log `MODEL_LOAD_FAILED: model directory not found: /app/models/churn-v2` + the ConfigMap value in use | ConfigMap fixed, then agent `restart_deployment` (confirmed) → healthy |
| Release with non-existent image `churn-api:1.1.0` | **Still serving** from 2 old pods (`maxUnavailable: 0`), but the new pod is in `ImagePullBackOff`. Image-pull failure with the registry error as evidence | Agent `rollback_deployment` (confirmed), revision 3 → 2, `churn-api:1.0.0` restored |

## Task 5: Multi-tool agent

`agent/orchestrator.py` runs a function-calling loop with **Gemini** (`GEMINI_MODEL`, default
`gemini-flash-lite-latest`) over all 36 tools. Each tool's pydantic input model becomes the
function declaration (structured inputs), and each result comes back as a typed `ToolResult`
(structured outputs). No agent framework is used; see [architecture.md](architecture.md) §8.

**Guarantees enforced in code (not just the prompt)**

| Guarantee | How |
|---|---|
| No invented results | a failed tool returns `data: null`; the LLM is told the fact is UNKNOWN; the orchestrator appends a **Data gaps** section to the answer regardless of what the LLM writes |
| No automatic state changes | a confirmation-required tool call is **parked** as a pending action; only `POST /agent/confirm {action_id}` runs it, once, with the exact arguments shown |
| Multi-system questions fully covered | before accepting an answer, a coverage check compares the systems the question mentions with the modules actually queried, and sends the model back once if one is missing |
| Retraining recommended only on evidence | `get_retraining_policy` (data freshness, model age, failed quality gate, supplied degradation); each rule is evaluated as triggered / not triggered / unknown |
| Bounded cost | step and tool-call budgets, identical-call cache, tool output truncation |

**Interfaces**

| Endpoint | Purpose |
|---|---|
| `POST /agent/query` | `{message, session_id?, include_evidence?}` → answer, tool trace, pending actions, data gaps |
| `GET /agent/pending` | actions awaiting confirmation |
| `POST /agent/confirm` | `{action_id}` → executes that action once |
| `GET /tools` | tool catalogue with JSON schemas and permission flags |
| `GET /health` | API liveness |

Set `AGENT_API_KEY` to require an `X-API-Key` header. Every response carries `X-Request-ID`.

### Demonstration scenarios (`python scripts/run_demo_conversations.py`, API must be running)

| Scenario | Tools the agent chose | Outcome (`docs/conversations/`) |
|---|---|---|
| Dataset provenance | get_deployed_version, get_model_versions, get_dataset_version, explain_dataset_reproduction | champion v1 → dataset **v1.1** (md5 verified), exact reproduction commands |
| Model selection | compare_model_runs, get_dag_runs, diagnose_dag_run | last 5 runs tied on F1 0.497 → picks the one that passed the gate; notes ties and unverified items |
| Pipeline execution | trigger_training_pipeline → **pending** → confirmed | real Airflow run: 6/6 tasks succeeded, new candidate registered |
| Deployment diagnosis (live outage injected) | diagnose_service | **unavailable**, 0 endpoints; `MODEL_LOAD_FAILED` + wrong `MODEL_DIR` in ConfigMap; fix ConfigMap then restart (rollback flagged as ineffective) |
| Multi-tool reasoning (mandatory) | find_best_model, get_model_versions, get_deployed_version, diagnose_service, get_dag_runs, get_retraining_policy, get_dataset_version ×2, get_dataset_history | champion trained on v1.1, latest data v2.0 → **DATA_FRESHNESS triggered**; a gate-passing v2.0 candidate exists, so **review/promote the candidate, don't retrain**. Nothing was triggered |
