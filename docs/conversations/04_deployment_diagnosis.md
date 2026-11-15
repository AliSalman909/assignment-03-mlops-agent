# Deployment diagnosis

**User:** Why is my prediction API unavailable?

_request `3d9d89d63e96` · model `gemini-flash-lite-latest` · 2 step(s)_

| step | tool | arguments | status | ms |
|---|---|---|---|---|
| 1 | `diagnose_service` | `{}` | ok | 973 |

**Agent:**

## Known facts
- **Service Status:** The `churn-api` service is **unavailable** with 0 ready endpoints (`diagnose_service`).
- **Pod Health:** Pods (`churn-api-79d85c4846-htl87`, `churn-api-79d85c4846-pltc4`) are running but failing their readiness/startup probes (`diagnose_service`).
- **Root Cause / Error:** The application fails to load the model because of a `FileNotFoundError`: the model directory specified does not exist (`/app/models/churn-v2/mlflow_model`) (`diagnose_service`).
- **Configuration Issue:** The ConfigMap (`churn-api-config`) sets `MODEL_DIR` to `/app/models/churn-v2`, whereas the model directory baked into the image is located at `/app/model` (`diagnose_service`).
- **Rollback Effectiveness:** A deployment rollback would **not** fix this issue because ConfigMaps are not part of deployment revisions (`diagnose_service`).

## Uncertainties
- None regarding the immediate service unavailability or configuration error.

## Recommended actions
1. Correct the value of `MODEL_DIR` in the `churn-api-config` ConfigMap to point to the correct model path baked into the image (`/app/model`).
2. Perform a rolling restart of the `churn-api` deployment (`restart_deployment`) so the pods reload the corrected configuration (note: requires explicit user confirmation).

