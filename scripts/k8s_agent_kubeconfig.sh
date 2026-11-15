#!/usr/bin/env bash
# Create a short-lived kubeconfig for the agent's restricted ServiceAccount (kubernetes/agent-rbac.yaml).
# The file is written to .secrets/ (git-ignored). Point the agent at it with K8S_KUBECONFIG.
set -euo pipefail
cd "$(dirname "$0")/.."

CTX="${KUBE_CONTEXT:-kind-mlops}"
OUT=".secrets/agent.kubeconfig"
mkdir -p .secrets

SERVER=$(kubectl config view --raw -o jsonpath="{.clusters[?(@.name==\"$CTX\")].cluster.server}")
CA=$(kubectl config view --raw -o jsonpath="{.clusters[?(@.name==\"$CTX\")].cluster.certificate-authority-data}")
TOKEN=$(kubectl --context "$CTX" -n mlops create token mlops-agent --duration=24h)

cat > "$OUT" <<EOF
apiVersion: v1
kind: Config
clusters:
- name: mlops
  cluster:
    server: $SERVER
    certificate-authority-data: $CA
users:
- name: mlops-agent
  user:
    token: $TOKEN
contexts:
- name: mlops-agent
  context: {cluster: mlops, user: mlops-agent, namespace: mlops}
current-context: mlops-agent
EOF
chmod 600 "$OUT"
echo "wrote $OUT (token valid 24h). Set K8S_KUBECONFIG=$OUT in .env"
