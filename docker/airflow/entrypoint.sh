#!/usr/bin/env bash
# Start Airflow 3 in standalone mode (api-server, scheduler, dag-processor, triggerer) with the
# Simple Auth Manager. The admin password comes from the environment (.env), never from the image.
set -euo pipefail

: "${AIRFLOW_ADMIN_PASSWORD:?AIRFLOW_ADMIN_PASSWORD must be set (see .env.example)}"

python - <<'EOF'
import json, os
path = os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"]
with open(path, "w") as fh:
    json.dump({"admin": os.environ["AIRFLOW_ADMIN_PASSWORD"]}, fh)
os.chmod(path, 0o600)
EOF

exec airflow standalone
