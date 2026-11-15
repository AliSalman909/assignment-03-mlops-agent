#!/usr/bin/env bash
# Exercise the MLflow + Airflow tools against the running docker compose stack and export
# the outputs for the report, including three deliberately injected pipeline failures.
#   docker compose up -d && bash scripts/collect_mlflow_airflow_evidence.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv/Scripts ]; then export PATH="$PWD/.venv/Scripts:$PATH"; else export PATH="$PWD/.venv/bin:$PATH"; fi
export MLFLOW_DISABLE_AGENT_HINT=1

M="docs/evidence/mlflow"
A="docs/evidence/airflow"
mkdir -p "$M" "$A"

run() { local out="$1"; shift; python -m agent.cli "$@" > "$out.json" 2>/dev/null || true
        python -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['tool']:26} {d['status']:22} {sys.argv[1]}\")" "$out.json"; }

# ---------------------------------------------------------------- MLflow (Task 2)
run "$M/01_list_experiments" list_experiments
run "$M/02_list_runs_recent" list_training_runs '{"limit": 10}'
run "$M/03_find_best_run_f1" find_best_model '{"metric": "f1"}'
run "$M/04_compare_last_5_runs_f1" compare_model_runs '{"last_n": 5, "metric": "f1"}'
run "$M/05_compare_experiments" compare_experiments '{"experiment_ids": ["churn-baseline", "churn-training-pipeline"]}'
run "$M/06_model_registry" get_model_versions '{"model_name": "churn-classifier"}'
BEST=$(python -c "import json; print(json.load(open('$M/03_find_best_run_f1.json'))['data']['best_run']['run_id'])")
run "$M/07_run_details_best" get_run_details "{\"run_id\": \"$BEST\"}"

# ---------------------------------------------------------------- Airflow (Task 3)
run "$A/01_list_dags" list_dags
run "$A/02_trigger_refused_without_confirmation" trigger_training_pipeline '{}'
run "$A/03_trigger_unapproved_dag_forbidden" trigger_training_pipeline '{"dag_id": "some_other_dag"}' --confirm

trigger() { local out="$1" conf="$2"
  python -m agent.cli trigger_training_pipeline "{\"config\": $conf, \"wait_seconds\": 600}" --confirm > "$out.json" 2>/dev/null || true
  python -c "import json,sys; d=json.load(open(sys.argv[1])); x=d['data'] or {}; print(f\"trigger_training_pipeline                {d['status']:8} run={x.get('state')} {sys.argv[1]}\")" "$out.json"; }

trigger "$A/04_run_success" '{}'
run "$A/05_diagnose_success" diagnose_dag_run

trigger "$A/06_FAILURE1_run_corrupt_data" '{"inject_failure": "corrupt_data"}'
run "$A/07_FAILURE1_failed_tasks" get_failed_tasks
run "$A/08_FAILURE1_validate_logs" get_task_logs '{"task_id": "validate_dataset", "max_lines": 40}'
run "$A/09_FAILURE1_diagnosis" diagnose_dag_run

trigger "$A/10_FAILURE2_run_quality_gate" '{"min_f1": 0.60}'
run "$A/11_FAILURE2_diagnosis" diagnose_dag_run

trigger "$A/12_FAILURE3_run_transient_retry" '{"inject_failure": "transient_error"}'
run "$A/13_FAILURE3_diagnosis" diagnose_dag_run

run "$A/14_recent_runs" get_dag_runs '{"limit": 10}'
run "$M/08_model_registry_after_pipeline" get_model_versions '{"model_name": "churn-classifier"}'
