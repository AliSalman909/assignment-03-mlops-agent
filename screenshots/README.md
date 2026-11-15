# Screenshots

Visual evidence for the technical report ("Screenshots or exported evidence from DVC, MLflow, Airflow,
and Kubernetes"). Save each file here with the name shown. Machine-readable evidence for the same
views is already exported in [`docs/evidence/`](../docs/evidence/) and [`docs/conversations/`](../docs/conversations/).

Start everything first: `python start.py --setup-only`

## DVC (terminal)

| File | Show | Command |
|---|---|---|
| `dvc_01_versions.png` | the three dataset versions as git tags | `git log --oneline --decorate -- data/raw/churn.csv.dvc` |
| `dvc_02_status.png` | data in sync with the record and the remote | `dvc status` and `dvc status --cloud` |
| `dvc_03_agent_provenance.png` | the agent resolving a dataset version | `python -m agent.cli get_dataset_version '{"git_commit": "v1.1"}'` |

## MLflow (http://localhost:5000)

| File | Show |
|---|---|
| `mlflow_01_experiments.png` | experiments `churn-baseline` and `churn-training-pipeline` with their runs |
| `mlflow_02_run_compare.png` | several runs selected → **Compare** (F1, ROC-AUC, precision, recall) |
| `mlflow_03_run_provenance.png` | one run's **Tags**: `git_commit`, `dataset_md5`, `quality_gate` |
| `mlflow_04_registry.png` | **Models → churn-classifier**: versions with the `champion` and `candidate` aliases |

## Airflow (http://localhost:8080, user `admin`, password in `.env`)

| File | Show |
|---|---|
| `airflow_01_dag_graph.png` | `churn_training_pipeline` **Graph**: validate → preprocess → train → evaluate → gate → register |
| `airflow_02_runs.png` | run history with successful and failed runs |
| `airflow_03_failed_validation.png` | a failed `validate_dataset` task log (`DATA_VALIDATION_FAILED`) |
| `airflow_04_quality_gate.png` | a failed `quality_gate` with `register_candidate` = upstream_failed |
| `airflow_05_retry.png` | `preprocess_data` with 2 attempts (transient failure, then success) |

To produce the failed runs: `bash scripts/collect_mlflow_airflow_evidence.sh` (Windows: run it in Git Bash).

## Kubernetes (terminal)

| File | Show | Command |
|---|---|---|
| `k8s_01_resources.png` | Deployment (2/2), pods, ClusterIP Service, ConfigMap | `kubectl -n mlops get deploy,pods,svc,cm -o wide` |
| `k8s_02_probes_resources.png` | probes and resource requests/limits | `kubectl -n mlops describe deploy churn-api` |
| `k8s_03_outage.png` | pods running but not ready during the injected outage | `bash scripts/k8s_inject_failure.sh broken-config`, then `kubectl -n mlops get pods` |
| `k8s_04_rbac.png` | the agent identity's permissions | `kubectl --kubeconfig .secrets/agent.kubeconfig auth can-i --list -n mlops` |

## Agent

| File | Show |
|---|---|
| `agent_01_mandatory_scenario.png` | `python start.py`, the mandatory multi-tool question and the answer (Known facts / Uncertainties / Recommended actions) |
| `agent_02_confirmation.png` | "Trigger the model training pipeline…" → the confirmation prompt → result |
| `agent_03_api_docs.png` | `python start.py --api` → http://localhost:8100/docs |
| `ci_01_github_actions.png` | a green GitHub Actions run (after pushing) |

Remember to restore the cluster after outage screenshots: `bash scripts/k8s_inject_failure.sh restore`.
