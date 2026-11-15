# Technical Report: Production-Ready MLOps AI Agent

**Assignment 03, MLOps & Production AI Engineering, Fall 2026**

This report documents an AI agent that answers operational questions about a customer-churn ML system
using evidence from real tools: **DVC** (dataset versions), **MLflow** (experiments and model registry),
**Apache Airflow 3** (training pipeline) and **Kubernetes** (the deployed prediction API). Every
operational claim in an answer comes from a tool call made during that conversation. All numbers in
this report come from the exported evidence in [`docs/evidence/`](docs/evidence/) and
[`docs/conversations/`](docs/conversations/), which can be regenerated with the scripts in `scripts/`.

---

## 1. System architecture

```
 Engineer ──HTTP / CLI──► FastAPI interface (api/main.py)        input validation, API key, request ids
                                   │
                                   ▼
                         Orchestrator (agent/orchestrator.py)     LLM loop, budgets, coverage check,
                                   │  ▲                           pending-action confirmation, data-gap disclosure
                    function calls │  │ structured results
                                   ▼  │
                         Gemini (agent/llm.py)                    gemini-flash-lite-latest → gemini-3.5-flash-lite
                                   │
                         Tool registry (agent/tools/base.py)      pydantic validation, permission policy,
                                   │                              typed ToolResult, JSON logs
        ┌──────────────┬───────────┼──────────────┬───────────────┐
        ▼              ▼           ▼              ▼               ▼
   DVC tools      MLflow tools  Airflow tools  Kubernetes tools  Retraining policy
   git + dvc      MlflowClient  REST API v2    official client
   (allow-list)                 + JWT          + restricted RBAC
        │              │           │              │
   git repo +     MLflow 3.18   Airflow 3.4    kind cluster, namespace mlops
   DVC remote     (Docker)      (Docker)       churn-api: 2 replicas, ClusterIP
                       ▲           │
                       └───────────┘  pipeline logs runs and registers candidate models
```

| Layer | Technology | Implementation |
|---|---|---|
| Interface | FastAPI (`/agent/query`, `/agent/confirm`, `/agent/pending`, `/tools`, `/health`) and an interactive CLI | `api/main.py`, `agent/main.py` |
| Agent | Google Gemini, manual function calling (no agent framework) | `agent/orchestrator.py`, `agent/llm.py` |
| Data versioning | DVC 3.67, git tags `data-v1.0`, `data-v1.1`, `data-v2.0`, DVC pipeline stage `preprocess` | `dvc.yaml`, `data/raw/churn.csv.dvc` |
| Experiment tracking | MLflow 3.18 server; experiments `churn-baseline`, `churn-training-pipeline`; registry `churn-classifier` | `docker-compose.yml`, `scripts/seed_experiments.py` |
| Orchestration | Airflow 3.4, DAG `churn_training_pipeline` | `dags/training_pipeline.py` |
| Serving | FastAPI inference service on Kubernetes (kind) | `Dockerfile`, `serving/`, `kubernetes/` |
| Setup | One command: `python start.py` (idempotent) | `start.py` |

**Request flow.** The API validates a request and passes it to the orchestrator with a request id. Gemini
receives a system prompt plus 36 function declarations, generated automatically from each tool's
pydantic input model. It issues tool calls, often several in parallel. Each call passes through the
registry, which validates arguments, enforces permissions, runs the tool, converts any failure into a
typed status and writes one JSON log line. Results go back to the model as `{status, data, error,
evidence_source}`. The loop ends when the model writes an answer. Before that answer is accepted, a
**coverage check** confirms that every system the question mentions was actually queried; otherwise
the model is sent back once to finish.

**Structured tool contract.** Every tool returns:

```python
ToolResult(tool, status, request_id, duration_ms, data, error, evidence_source)
# status: ok | error | unavailable | timeout | invalid_request | confirmation_required | forbidden
# data is always null unless status == "ok", so a failure can never be mistaken for evidence
```

**Data provenance.** Every MLflow run is tagged with the `git_commit` and `dataset_md5` of its training
data. The DVC tool resolves those tags to a dataset version by **content hash**: it compares the md5
against the hash recorded at each `data-v*` tag, so labels can't drift from the data they describe.

**Training pipeline.** `validate_dataset → preprocess_data → train_model → evaluate_and_log → quality_gate →
register_candidate`. Each task retries twice (10 s apart). Data-validation and quality-gate failures raise
`AirflowFailException`, which skips retries. The gate requires F1 ≥ 0.48, ROC-AUC ≥ 0.70, and at most
0.02 F1 regression against the current champion. A model that passes is registered with the `candidate`
alias only; promoting it to `champion` (the served model) is a human decision.

**Serving.** The `churn-api` Deployment runs 2 replicas (rolling update with `maxUnavailable: 0`):
- **Probes:** startup, readiness (= model loaded) and liveness.
- **Resources:** requests 100m CPU / 256 Mi, limits 500m / 512 Mi.
- **Security:** non-root user, all Linux capabilities dropped.
- **Configuration and networking:** settings from a ConfigMap; exposed through a ClusterIP Service.

The image contains the registry's `champion` model, exported with its lineage; the export refuses to run
if DVC can't verify the model's training data.

![Kubernetes resources](screenshots/k8s_01_resources.png)
![Probes and resource limits](screenshots/k8s_02_probes_resources.png)

More detail: [architecture.md](architecture.md).

---

## 2. Tool definitions and permissions

36 tools in five modules. Tools marked **⚠** change state and require explicit user confirmation.

| Module | Tools | Interface |
|---|---|---|
| **DVC** (8) | `list_datasets`, `get_dataset_version`, `get_dataset_history`, `compare_dataset_versions`, `check_dataset_changes`, `explain_dataset_reproduction`, `reproduce_dataset` (writes only to a sandbox folder), ⚠ `restore_dataset_workspace` | allow-listed `git` / `dvc` subprocess (`shell=False`), DVC Python API |
| **MLflow** (7) | `list_experiments`, `list_training_runs`, `get_run_details`, `find_best_model`, `compare_model_runs`, `compare_experiments`, `get_model_versions` | `MlflowClient` (read-only) |
| **Airflow** (7) | `list_dags`, `get_dag_runs`, `get_dag_run_status`, `get_failed_tasks`, `get_task_logs`, `diagnose_dag_run`, ⚠ `trigger_training_pipeline` | REST API v2, JWT from `/auth/token` |
| **Kubernetes** (13) | `list_pods`, `list_deployments`, `get_pod_status`, `get_pod_logs`, `list_services`, `inspect_service`, `find_unhealthy_pods`, `get_events`, `get_deployed_version`, `diagnose_service`, ⚠ `restart_deployment`, ⚠ `scale_deployment`, ⚠ `rollback_deployment` | official Python client |
| **Policy** (1) | `get_retraining_policy`: explicit rules (data freshness, model age, quality-gate failure, production degradation) | configuration |

**Permission model: four layers**

1. **No self-approval.** When the LLM calls a ⚠ tool, the call is not executed. It is stored as a pending
   action (id, tool, exact arguments, 15-minute expiry) and shown to the user. Only `POST /agent/confirm
   {action_id}`, or typing `yes` in the CLI, executes it: once, with exactly those arguments.
2. **Allow-lists inside the tools.**
   - **Airflow:** only DAGs in `AIRFLOW_APPROVED_DAGS`, and only known `config` keys.
   - **Kubernetes:** only namespaces in `K8S_APPROVED_NAMESPACES` (`mlops`), only Deployments labelled
     `app.kubernetes.io/part-of=mlops-agent`, at most 5 replicas.
   - **DVC:** only allow-listed sub-commands; revisions and paths are regex-validated (no option
     injection, no `..`, no absolute paths).
3. **Least-privilege credentials.** The agent's Kubernetes ServiceAccount
   ([kubernetes/agent-rbac.yaml](kubernetes/agent-rbac.yaml)) may read the application's resources and
   patch Deployments in `mlops` only. Verified with `kubectl auth can-i`: it can't delete pods, read
   Secrets, or see `kube-system`. The Airflow container mounts the project read-only.
4. **Secret handling.** Secrets live only in the git-ignored `.env` and `.secrets/`. Logs, tool
   outputs, and ConfigMap values shown to the LLM are redacted (bearer tokens, JWTs, API keys, URL
   credentials, `password=` patterns). `scripts/check_secrets.py` runs before each push.

![Agent RBAC](screenshots/k8s_04_rbac.png)

**Observability.** One JSON log line per event, without secrets:
- `tool_call`: tool, status, duration, request id;
- `llm_call`: model, latency, number of function calls, token counts;
- `agent_request` / `agent_response`, `action_confirmed`, `http_request`.

---

## 3. Evidence from DVC, MLflow, Airflow and Kubernetes

### DVC

Three dataset versions, each a git tag whose `.dvc` file records the content hash. The bytes are stored in
the DVC remote ([docs/evidence/dvc/02_get_dataset_history.json](docs/evidence/dvc/02_get_dataset_history.json)):

| Version | Commit | md5 | Rows | Change |
|---|---|---|---|---|
| `data-v1.0` | `4ebc9ee` | `43d9a0e1…` | 4,040 | initial data (contains duplicates) |
| `data-v1.1` | `0f4a619` | `45d0f04a…` | 6,060 | +2,020 customers |
| `data-v2.0` | `e8f5aaf` | `9f6600f3…` | 6,000 | new feature `last_login_days`, deduplicated |

`compare_dataset_versions(v1.0, v2.0)` reported +1,960 rows, the added column `last_login_days`, and a
positive rate change from 0.2653 to 0.2298. `reproduce_dataset(v1.0)` downloaded v1.0 from the remote and
verified its md5 (`43d9a0e1…`, `verified: true`).

![DVC versions](screenshots/dvc_01_versions.png)
![DVC status](screenshots/dvc_02_status.png)

### MLflow

- **Experiment `churn-baseline`:** 9 runs, three model families trained on each dataset version. Every
  run carries `git_commit` and `dataset_md5` tags.
- **Experiment `churn-training-pipeline`:** the runs created by Airflow.
- **Registry `churn-classifier`:** version 1 holds the `champion` alias (logistic regression trained on
  v1.1: F1 0.4707, ROC-AUC 0.6673). Gate-passing pipeline versions hold the `candidate` alias.

![MLflow experiments](screenshots/mlflow_01_experiments.png)
![MLflow run comparison](screenshots/mlflow_02_run_compare.png)
![MLflow provenance tags](screenshots/mlflow_03_run_provenance.png)
![MLflow model registry](screenshots/mlflow_04_registry.png)

### Airflow

The DAG was run through the agent: one normal run (6/6 tasks succeeded) and three runs with injected
failures, described in §5 and in [docs/evidence/airflow/](docs/evidence/airflow/).

![Airflow DAG graph](screenshots/airflow_01_dag_graph.png)
![Airflow runs](screenshots/airflow_02_runs.png)

### Kubernetes

`churn-api` in namespace `mlops`: 2/2 replicas ready, a ClusterIP Service with 2 ready endpoints, and
`/version` reporting model v1 (`champion`) trained on dataset v1.1
([docs/evidence/kubernetes/](docs/evidence/kubernetes/)).

---

## 4. Five tested agent conversations

Run through the real HTTP API against the live services (`scripts/run_demo_conversations.py`). The full
transcripts, tool traces and raw JSON are in [docs/conversations/](docs/conversations/).

| # | Scenario | Tools the agent chose (all `ok`) | Result |
|---|---|---|---|
| 1 | **Dataset provenance**: *"Which dataset version was used to train the model that is currently deployed? Verify it against DVC and explain how I can reproduce it."* | `get_deployed_version`, `get_model_versions`, `get_dataset_version`, `explain_dataset_reproduction` | Champion v1 → dataset **v1.1** (commit `0f4a619`, md5 verified against DVC). Gave the exact `git checkout` / `dvc pull` / `dvc get` commands and confirmed the bytes are in the cache and the remote. |
| 2 | **Model selection**: *"Compare the last five training runs and recommend the best candidate based on F1 score."* | `compare_model_runs`, `get_dag_runs`, `diagnose_dag_run` | The five most recent runs are tied at F1 0.497 (same data, same deterministic model). The agent chose the one that **passed the quality gate**, stated the tie, and listed what was still unverified rather than treating top F1 as production-ready. |
| 3 | **Pipeline execution**: *"Trigger the model training pipeline and report whether execution succeeded."* | `trigger_training_pipeline` → **pending** → user confirmed → `trigger_training_pipeline` | Not executed until confirmed. After confirmation: a real Airflow run, 6/6 tasks succeeded in ~25 s, new candidate registered. |
| 4 | **Deployment diagnosis**: *"Why is my prediction API unavailable?"* (live outage injected) | `diagnose_service` | **Unavailable, 0 ready endpoints.** Root cause: model failed to load because `MODEL_DIR` in the ConfigMap points to a missing path. Recommended: fix the ConfigMap, then restart (confirmation required). Flagged rollback as ineffective. See §5. |
| 5 | **Multi-tool reasoning (mandatory)**: *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."* | `find_best_model`, `get_model_versions`, `get_deployed_version`, `diagnose_service`, `get_dag_runs`, `get_retraining_policy`, `get_dataset_version` ×2, `get_dataset_history` | See below. |

**Scenario 5 in detail.** The agent queried all four systems and the policy, then answered in the
required Known facts / Uncertainties / Recommended actions format:

- **Known facts**
  - **Best run:** `cedfcc5e…` (logistic regression, F1 0.5136, ROC-AUC 0.7513), trained on dataset
    **v2.0**, verified through DVC by md5.
  - **Deployment:** healthy (2/2 replicas), serving `champion` v1, which was trained on **v1.1**.
  - **Latest data:** v2.0 is the latest version according to `get_dataset_history`.
  - **Pipeline:** the most recent pipeline runs succeeded.
- **Policy evaluation**
  - `DATA_FRESHNESS`: **triggered** (the champion has never seen v2.0).
  - `MODEL_AGE`: **not triggered**.
  - `QUALITY_GATE_FAILURE`: **not triggered**.
  - `PRODUCTION_DEGRADATION`: **unknown** (no production metrics were supplied).
- **Uncertainties:** live production performance is not measured.
- **Recommendation:** **don't retrain.** A `candidate` trained on v2.0 has already passed the quality
  gate, so it should be reviewed by a human and promoted to `champion`, then rolled out. **Nothing was
  triggered**, as the assignment requires.

![Mandatory scenario](screenshots/agent_01_mandatory_scenario.png)
![Confirmation flow](screenshots/agent_02_confirmation.png)

---

## 5. Failure investigations

### Investigation 1: Prediction API outage caused by a configuration change (Kubernetes)

**Symptom.** An operator changed `MODEL_DIR` in the `churn-api-config` ConfigMap to
`/app/models/churn-v2`, and the pods were recycled. Asked *"Why is my prediction API unavailable?"*, the
agent called `diagnose_service`, which correlates the Service, its endpoints, pods, events, logs and
configuration.

**Evidence collected by the agent** ([10_FAILURE_A_diagnose.json](docs/evidence/kubernetes/10_FAILURE_A_diagnose.json)):
- **Service:** 0 ready endpoints, so status **unavailable**.
- **Pods:** both `Running` but not ready: the readiness probe was failing.
- **Event:** `Readiness probe failed: HTTP probe failed with statuscode: 503`.
- **Log:** `MODEL_LOAD_FAILED FileNotFoundError: model directory not found: /app/models/churn-v2/mlflow_model`.
- **Configuration in use:** `MODEL_DIR=/app/models/churn-v2`. The image contains the model at `/app/model`.

**Root cause.** A configuration error in the application layer. The container started, but couldn't find
its model at the configured path, so `/health/ready` returned 503. Kubernetes correctly removed both pods
from the Service, which left no endpoints.

**Resolution.**
1. The ConfigMap was restored (`MODEL_DIR=/app/model`).
2. The agent requested `restart_deployment`, which waited for confirmation and ran once confirmed.
3. Pods were replaced one at a time; `diagnose_service` then reported *"service is healthy: 2 ready
   endpoint(s), readiness check passed"*.

The diagnosis states that a rollback would **not** help: ConfigMaps aren't part of a Deployment revision,
so rolled-back pods would read the same broken value.

**What made this diagnosable.** The readiness probe reflects whether the model is loaded, not just
whether the process is up. The application logs a distinct `MODEL_LOAD_FAILED` marker. The tool returns
the ConfigMap values it found (with sensitive keys redacted).

![Outage](screenshots/k8s_03_outage.png)

### Investigation 2: Training run blocked by invalid input data (Airflow)

**Symptom.** A pipeline run (`manual__2026-11-15T11:17:06…`) was triggered with an upstream export that
had lost its label column (injected via `inject_failure=corrupt_data`). The agent called
`get_failed_tasks`, `get_task_logs` and `diagnose_dag_run`.

**Evidence collected by the agent** ([09_FAILURE1_diagnosis.json](docs/evidence/airflow/09_FAILURE1_diagnosis.json)):
- **Failed task:** `validate_dataset`, attempt 1 of 3 (no retries: the failure is deterministic).
- **Log:** `AirflowFailException: DATA_VALIDATION_FAILED: required_columns: missing=['churned']; duplicate_rate: duplicate rows=11.765% (max 5%)`.
- **Blocked tasks:** `preprocess_data`, `train_model`, `evaluate_and_log`, `quality_gate` and
  `register_candidate` are all `upstream_failed`, so `promotion_blocked: true`.

**Root cause.** The data layer. The input was missing the target column and had 11.8% duplicate rows.
Validation caught it before any training happened.

**Resolution.** No model was trained, logged or registered. The required "a failed validation must
prevent inappropriate downstream promotion" is enforced by task dependencies, not by convention. The
diagnosis's recommended action was to inspect the dataset version with the DVC tools, then fix or roll back
the data and re-run. The versioned dataset itself was intact: `check_dataset_changes` reports the
workspace matching its committed DVC version
([05_check_dataset_changes_clean.json](docs/evidence/dvc/05_check_dataset_changes_clean.json)). So the fix
is to re-run from the DVC-tracked data rather than the broken export. Later runs on that data succeeded and
registered a candidate (conversation 3).

![Failed validation](screenshots/airflow_03_failed_validation.png)

### Other injected failures (all diagnosed correctly)

| Injection | Agent diagnosis | Outcome |
|---|---|---|
| Tracked dataset edited without versioning (DVC) | `check_dataset_changes`: tracked data differs from the committed version; `preprocess` stage stale | `restore_dataset_workspace` refused without confirmation; after confirmation, data restored and md5 verified |
| Quality gate threshold `min_f1=0.60` (Airflow) | `quality_gate` failed, **model layer**: `f1 0.4970 < min_f1 0.6` | `register_candidate` never ran; no promotion |
| Transient storage timeout (Airflow) | `preprocess_data` failed on attempt 1 and succeeded on retry, **infrastructure (recovered)** | Retry policy absorbed the fault; run succeeded |
| Release with non-existent image `churn-api:1.1.0` (Kubernetes) | New pod in `ImagePullBackOff`, registry error as evidence; service **still available** on the 2 old pods (`maxUnavailable: 0`) | `rollback_deployment` (confirmed): revision 3 → 2, `churn-api:1.0.0` restored |

---

## 6. Testing, reliability and safety

76 automated tests (`pytest`). They run in GitHub Actions on every pull request, on Python 3.12 and 3.13,
with lint, format and `pip check` steps. During CI, MLflow, Airflow and Kubernetes are pointed at
unreachable addresses, which proves the suite never touches real infrastructure.

The suite covers the brief's required cases:
- **Tool selection and execution:** multi-tool requests run the selected tools and pass structured
  results back to the model.
- **Failures:** timeouts, invalid requests, unknown tools, service failures, and an LLM outage.
- **No invented results:** when a tool is unavailable, it returns `data: null` and the answer carries a
  "Data gaps" section, even if the LLM writes something else.
- **Unauthorized Kubernetes changes are rejected:** no confirmation, a namespace outside the allow-list,
  an unlabelled Deployment, or too many replicas; the test also asserts nothing was patched.
- **Confirmation rules:** a confirmation is single-use, and a retraining question never triggers training.
- **Logging:** structured logs carry tool, status, duration and request id, without secrets.

Before release, the code also had an independent review and was re-run against the live services. That
fixed nine defects, including session-history trimming that could break long conversations, and a
rollback that merged the old template instead of replacing it.

---

## 7. Known limitations and proposed improvements

| Limitation | Proposed improvement |
|---|---|
| Sessions and pending actions are held in memory, and confirmations aren't tied to a user identity | Store them in Redis or a database; authenticate users (OIDC) and record who confirmed each action |
| `PRODUCTION_DEGRADATION` can only be evaluated from metrics a user supplies | Add monitoring (e.g. Evidently or Prometheus) that logs live performance to MLflow, so the rule can be evaluated automatically |
| Demo-grade infrastructure: local DVC remote committed to git, SQLite-backed MLflow, Airflow standalone, single-node kind | Object storage (S3/GCS) for DVC and artifacts, Postgres for MLflow and Airflow, CeleryExecutor or KubernetesExecutor, a managed cluster |
| Promotion to `champion` and rollout of the new model are manual | A confirmed `promote_model` tool, plus GitOps (e.g. Argo CD) to roll out the new image |
| The agent's Kubernetes token is short-lived (24 h) and regenerated locally | Workload identity or a secrets manager |
| A lightweight LLM can mis-state secondary details | The answer format requires a source tool for each fact, and raw evidence is available with `include_evidence=true`; a stronger model or a self-check pass would reduce this further |
| Some tool defaults are project-specific (DAG `churn_training_pipeline`, service `churn-api`) | Move them into `.env`; connection settings are already configurable, and `python start.py --external` targets other systems |

---

## Reproducing this report

```bash
python start.py --setup-only                               # full local stack
bash scripts/collect_dvc_evidence.sh                       # DVC evidence (incl. tampering)
bash scripts/collect_mlflow_airflow_evidence.sh            # MLflow + Airflow evidence (3 injected failures)
bash scripts/collect_k8s_evidence.sh                       # Kubernetes evidence (2 injected outages)
python start.py --api   &   python scripts/run_demo_conversations.py   # the five conversations
pytest -q                                                  # 76 tests
```

(On Windows, run the `.sh` scripts in Git Bash.)
