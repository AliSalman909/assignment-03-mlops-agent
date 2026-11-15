"""DVC pipeline stage: clean the raw churn data and split it into train/test sets.

Also writes `stats.json` (row counts, class balance, schema) so data changes are visible
through `dvc metrics` and to the agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "churned"


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df = df.drop(columns=["customer_id"])
    df = pd.get_dummies(df, columns=["contract_type", "internet_service"], dtype=int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/churn.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    df = preprocess(raw)
    train, test = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df[TARGET])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False, lineterminator="\n")
    test.to_csv(out / "test.csv", index=False, lineterminator="\n")

    stats = {
        "raw_rows": int(len(raw)),
        "duplicates_removed": int(len(raw) - len(raw.drop_duplicates())),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "n_features": int(df.shape[1] - 1),
        "positive_rate": round(float(df[TARGET].mean()), 4),
        "columns": list(raw.columns),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", newline="\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
