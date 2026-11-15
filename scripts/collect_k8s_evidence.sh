#!/usr/bin/env bash
# Exercise the Kubernetes tools against the kind cluster and export evidence, including two
# injected outages and their agent-driven resolution.
#   bash scripts/collect_k8s_evidence.sh
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv/Scripts ]; then export PATH="$PWD/.venv/Scripts:$PATH"; else export PATH="$PWD/.venv/bin:$PATH"; fi
K="kubectl --context ${KUBE_CONTEXT:-kind-mlops} -n mlops"

E="docs/evidence/kubernetes"
mkdir -p "$E"
rm -f docs/evidence/kubernetes/*.json

run() { local out="$1"; shift; python -m agent.cli "$@" > "$E/$out.json" 2>/dev/null || true
  python -c "import json,sys; d=json.load(open(sys.argv[1])); x=d['data'] or {}; print(f\"{d['tool']:22} {d['status']:22} {x.get('summary') or x.get('status') or ''}\")" "$E/$out.json"; }
pod_of() { $K get pods -l app.kubernetes.io/name=churn-api --sort-by=.metadata.creationTimestamp -o jsonpath="{.items[-1].metadata.name}"; }

echo "== healthy baseline"
run 01_list_pods list_pods
run 02_list_deployments list_deployments
run 03_list_services list_services
run 04_inspect_service inspect_service '{"service_name": "churn-api"}'
run 05_get_deployed_version get_deployed_version
run 06_diagnose_healthy diagnose_service '{"service_name": "churn-api"}'

echo "== safety controls"
run 07_restart_without_confirmation restart_deployment '{}'
run 08_restart_kube_system_forbidden restart_deployment '{"namespace": "kube-system", "deployment_name": "coredns"}' --confirm
run 09_list_pods_kube_system_forbidden list_pods '{"namespace": "kube-system"}'

echo "== FAILURE A: broken ConfigMap -> total outage"
bash scripts/k8s_inject_failure.sh broken-config
sleep 30
run 10_FAILURE_A_diagnose diagnose_service '{"service_name": "churn-api"}'
run 11_FAILURE_A_unhealthy_pods find_unhealthy_pods
run 12_FAILURE_A_pod_logs get_pod_logs "{\"pod_name\": \"$(pod_of)\", \"tail_lines\": 20}"
run 13_FAILURE_A_events get_events '{"warnings_only": true, "limit": 10}'
echo "-- resolution: operator fixes the ConfigMap, agent restarts the deployment (confirmed)"
$K apply -f kubernetes/configmap.yaml
run 14_FAILURE_A_restart_confirmed restart_deployment '{}' --confirm
$K rollout status deployment/churn-api --timeout=180s
run 15_FAILURE_A_after_fix diagnose_service '{"service_name": "churn-api"}'

echo "== FAILURE B: bad release image -> stalled rollout"
bash scripts/k8s_inject_failure.sh bad-image
sleep 40
run 16_FAILURE_B_diagnose diagnose_service '{"service_name": "churn-api"}'
run 17_FAILURE_B_deployments list_deployments
echo "-- resolution: agent rolls back (confirmed)"
run 18_FAILURE_B_rollback_confirmed rollback_deployment '{}' --confirm
$K rollout status deployment/churn-api --timeout=180s
run 19_FAILURE_B_version_after_rollback get_deployed_version
run 20_FAILURE_B_after_fix diagnose_service '{"service_name": "churn-api"}'
