#!/usr/bin/env bash
# Build and deploy the churn prediction API to the kind cluster.
#   1. export churn-classifier@champion (+ verified lineage) from MLflow
#   2. build the image   3. load it into kind   4. apply manifests   5. wait for rollout
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv/Scripts ]; then export PATH="$PWD/.venv/Scripts:$PATH"; else export PATH="$PWD/.venv/bin:$PATH"; fi
TAG="${APP_VERSION:-1.0.0}"
CLUSTER="${KIND_CLUSTER:-mlops}"
K="kubectl --context kind-$CLUSTER"

MLFLOW_DISABLE_AGENT_HINT=1 python serving/export_model.py --alias "${MODEL_ALIAS:-champion}"
docker build -t "churn-api:$TAG" .
kind load docker-image "churn-api:$TAG" --name "$CLUSTER"
$K apply -f kubernetes/namespace.yaml
$K apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml -f kubernetes/agent-rbac.yaml
$K -n mlops rollout status deployment/churn-api --timeout=180s
$K -n mlops get pods,svc -o wide
