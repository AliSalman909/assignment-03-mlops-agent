#!/usr/bin/env bash
# Run every DVC tool against the real repo and export the outputs for the report.
# Includes one injected failure (tampered dataset) and always restores the data on exit.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv/Scripts ]; then export PATH="$PWD/.venv/Scripts:$PATH"; else export PATH="$PWD/.venv/bin:$PATH"; fi

OUT="docs/evidence/dvc"
mkdir -p "$OUT"
rm -f docs/evidence/dvc/*.json

restore() { dvc checkout -q --force data/raw/churn.csv.dvc; }
trap restore EXIT

run() { local name="$1"; shift; python -m agent.cli "$@" > "$OUT/$name.json" 2>/dev/null || true; }

run 01_list_datasets list_datasets
run 02_get_dataset_history get_dataset_history
run 03_get_dataset_version_v1.1 get_dataset_version '{"git_commit": "v1.1"}'
run 04_compare_dataset_versions_v1.0_v2.0 compare_dataset_versions '{"version_a": "v1.0", "version_b": "v2.0"}'
run 05_check_dataset_changes_clean check_dataset_changes
run 06_explain_dataset_reproduction_v1.0 explain_dataset_reproduction '{"git_commit": "v1.0"}'
run 07_reproduce_dataset_v1.0 reproduce_dataset '{"git_commit": "data-v1.0"}'

# --- injected failure: someone edits tracked data without versioning it
echo "C999999,1,month-to-month,10,10,9,1,none,0,0,1" >> data/raw/churn.csv
run 08_FAILURE_tampered_data_detected check_dataset_changes
run 09_restore_refused_without_confirmation restore_dataset_workspace '{}'
run 10_restore_confirmed restore_dataset_workspace '{}' --confirm
run 11_check_after_restore check_dataset_changes

for f in "$OUT"/*.json; do
  python -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['tool']:30} {d['status']}\")" "$f"
done
