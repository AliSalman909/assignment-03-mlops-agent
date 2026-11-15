#!/usr/bin/env bash
# Deliberately break the prediction API (operator action with admin credentials) so the agent
# can diagnose it.  Usage: bash scripts/k8s_inject_failure.sh broken-config|bad-image|restore
set -euo pipefail
cd "$(dirname "$0")/.."
K="kubectl --context ${KUBE_CONTEXT:-kind-mlops} -n mlops"

case "${1:-}" in
  broken-config)
    # someone points MODEL_DIR at a path that doesn't exist, then the pods get recycled
    $K patch configmap churn-api-config --type merge -p '{"data":{"MODEL_DIR":"/app/models/churn-v2"}}'
    $K delete pods -l app.kubernetes.io/name=churn-api --wait=false
    echo "injected: ConfigMap MODEL_DIR -> /app/models/churn-v2 (missing) and pods recycled"
    ;;
  bad-image)
    # a release references an image tag that was never built/pushed
    $K set image deployment/churn-api churn-api=churn-api:1.1.0
    $K annotate deployment/churn-api kubernetes.io/change-cause="release churn-api 1.1.0" --overwrite
    echo "injected: deployment image -> churn-api:1.1.0 (does not exist)"
    ;;
  restore)
    $K apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml
    $K rollout restart deployment/churn-api
    $K rollout status deployment/churn-api --timeout=180s
    ;;
  *) echo "usage: $0 broken-config|bad-image|restore" >&2; exit 2 ;;
esac
