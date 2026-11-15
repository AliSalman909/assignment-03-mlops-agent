# Dataset provenance

**User:** Which dataset version was used to train the model that is currently deployed? Verify it against DVC and explain how I can reproduce that exact dataset.

_request `f316e559819a` · model `gemini-flash-lite-latest` · 3 step(s)_

| step | tool | arguments | status | ms |
|---|---|---|---|---|
| 1 | `get_deployed_version` | `{}` | ok | 918 |
| 1 | `get_model_versions` | `{}` | ok | 1855 |
| 2 | `get_dataset_version` | `{"git_commit": "0f4a61990e9ef14143179c22eed7d951bd571c9b", "dataset_path": "data/raw/churn` | ok | 269 |
| 2 | `explain_dataset_reproduction` | `{"dataset_path": "data/raw/churn.csv", "git_commit": "0f4a61990e9ef14143179c22eed7d951bd57` | ok | 142 |

**Agent:**

## Known facts
- **Currently deployed model**: `churn-classifier` version `1` (champion alias), run ID `d53d7934671e4824bb68d908cacae91d` (`GET deployed version`, `GET model versions`).
- **Dataset version**: Trained on dataset version **`v1.1`** (tag `data-v1.1`), path `data/raw/churn.csv`, with git commit `0f4a61990e9ef14143179c22eed7d951bd571c9b` and MD5 hash `45d0f04af574f1b29a3867bf1fcae39c` (`GET deployed version`, `GET model versions`).
- **DVC Verification**: Verified via `get_dataset_version` that commit `0f4a6199` points to `data/raw/churn.csv.dvc` with matching MD5 hash `45d0f04af574f1b29a3867bf1fcae39c` (`GET dataset version`).
- **Availability**: The dataset bytes for version `v1.1` are present in both the local DVC cache and remote storage (`GET dataset version`, `EXPLAIN dataset reproduction`).

## Uncertainties
- None regarding the dataset version or verification.

## Recommended actions
To reproduce this exact dataset (`v1.1`) without altering your current workspace, you can pull a non-destructive copy into the sandbox artifacts directory:
```bash
dvc get . data/raw/churn.csv --rev 0f4a6199 -o artifacts/reproduced/0f4a6199/churn.csv
```

Alternatively, to check it out directly in your workspace:
1. `git checkout 0f4a6199 -- data/raw/churn.csv.dvc`
2. `dvc pull data/raw/churn.csv.dvc`
3. `dvc status data/raw/churn.csv.dvc` (to verify)
4. (When done) `git checkout HEAD -- data/raw/churn.csv.dvc && dvc checkout data/raw/churn.csv.dvc`

