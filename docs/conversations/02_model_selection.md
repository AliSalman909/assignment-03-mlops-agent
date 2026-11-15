# Model selection

**User:** Compare the last five training runs and recommend the best candidate based on F1 score.

_request `2006858e376d` · model `gemini-flash-lite-latest` · 4 step(s)_

| step | tool | arguments | status | ms |
|---|---|---|---|---|
| 1 | `compare_model_runs` | `{"metric": "f1", "last_n": 5}` | ok | 331 |
| 3 | `get_dag_runs` | `{"limit": 5, "dag_id": "churn_training_pipeline"}` | ok | 235 |
| 3 | `diagnose_dag_run` | `{"dag_id": "churn_training_pipeline"}` | ok | 117 |

**Agent:**

## Known facts
- **Last 5 Training Runs (Source: `compare_model_runs`):**
  All five recent runs share identical evaluation metrics (`accuracy: 0.725`, `f1: 0.497`, `precision: 0.4289`, `recall: 0.5906`, `roc_auc: 0.7282`) and were trained on `4800` rows of `data/raw/churn.csv`.
  1. Run `4de440aea3ef44dbadfd8b1a9103cf4a` (Gradient Boosting, `quality_gate: passed`, git commit `0aaddf4ec0516a79a810007a03272e41bea3ee7c`, dataset md5 `9f6600f3d5da4d0ca84b02b9cb9199da`)
  2. Run `f6dd4b6b6739458b9ead84bb9e68c978` (Gradient Boosting, `quality_gate: failed`, git commit `0aaddf4ec0516a79a810007a03272e41bea3ee7c`, dataset md5 `9f6600f3d5da4d0ca84b02b9cb9199da`)
  3. Run `74563ab7f57e4eab9fe818735f9f50af` (Gradient Boosting, `quality_gate: passed`, git commit `0aaddf4ec0516a79a810007a03272e41bea3ee7c`, dataset md5 `9f6600f3d5da4d0ca84b02b9cb9199da`)
  4. Run `35824c47d8c44f0697cdfa88906dc8b8` (Gradient Boosting, `quality_gate: passed`, git commit `be063f5d8bac387fc0b11e48442d28a290aa1bd8`, dataset md5 `9f6600f3d5da4d0ca84b02b9cb9199da`)
  5. Run `db5f9b6305984215b8f5738ca07f9cd1` (Gradient Boosting, `quality_gate: failed`, git commit `be063f5d8bac387fc0b11e48442d28a290aa1bd8`, dataset md5 `9f6600f3d5da4d0ca84b02b9cb9199da`)

- **Airflow Training Pipeline Status (Source: `get_dag_runs` & `diagnose_dag_run`):**
  - The `churn_training_pipeline` DAG has 5 recent runs (3 successes, 2 failures).
  - The latest run (`manual__2026-11-15T11:17:41.435627+00:00`, matching run `4de440aea3ef44dbadfd8b1a9103cf4a`) succeeded with a retry on `preprocess_data` due to a transient object storage timeout (`TimeoutError: TRANSIENT: object storage read timed out (attempt 1)`).

## Uncertainties
- Dataset version labels/tags are unresolved in MLflow tags; DVC version verification has not been explicitly performed for these specific commits/md5s.
- Additional offline checks (e.g. latency, fairness, robustness) and business sign-off are not recorded in MLflow.

## Recommended actions
1. **Select Run `4de440aea3ef44dbadfd8b1a9103cf4a`** as the top candidate because it achieves the maximum F1 score (`0.497`, tied with all other recent runs), successfully passed the quality gate, and corresponds to a successfully completed Airflow pipeline run.
2. **Verify Dataset Provenance:** Run `get_dataset_version(git_commit='0aaddf4ec0516a79a810007a03272e41bea3ee7c', expected_md5='9f6600f3d5da4d0ca84b02b9cb9199da')` to confirm dataset integrity before promotion consideration.
3. **Review Candidate Registry Status:** Check whether this run is registered under the `candidate` alias in the MLflow Model Registry and evaluate whether it meets production readiness criteria beyond raw metrics.

