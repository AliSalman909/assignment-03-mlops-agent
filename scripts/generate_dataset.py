"""Deterministic generator for the customer-churn dataset used throughout the project.

Each dataset *version* is fully determined by this script + the version name, so the same
bytes (and therefore the same DVC md5) are produced on every machine. The history is:

  v1.0  4,000 rows, 8 features                     - initial data drop
  v1.1  6,000 rows, same schema                     - more customers collected
  v2.0  6,000 rows + `last_login_days` feature,     - new behavioural signal, duplicates removed
        dedup + label noise reduced

Usage:
    python scripts/generate_dataset.py --version v1.0 --output data/raw/churn.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONS: dict[str, dict] = {
    "v1.0": {"rows": 4000, "seed": 101, "extra_feature": False, "label_noise": 0.12, "dupes": 40},
    "v1.1": {"rows": 6000, "seed": 101, "extra_feature": False, "label_noise": 0.12, "dupes": 60},
    "v2.0": {"rows": 6000, "seed": 101, "extra_feature": True, "label_noise": 0.06, "dupes": 0},
}


def generate(version: str) -> pd.DataFrame:
    cfg = VERSIONS[version]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["rows"]

    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    monthly = np.round(rng.normal(65, 22, n).clip(15, 140), 2)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.05, n), 2)
    tickets = rng.poisson(1.4, n)
    products = rng.integers(1, 6, n)
    internet = rng.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.20])
    senior = (rng.random(n) < 0.16).astype(int)
    last_login = rng.exponential(12, n).round().astype(int)

    logit = (
        -1.2
        - 0.045 * tenure
        + 0.018 * (monthly - 65)
        + 0.35 * tickets
        - 0.25 * (products - 1)
        + np.where(contract == "month-to-month", 1.3, np.where(contract == "one-year", 0.1, -0.9))
        + np.where(internet == "fiber", 0.4, 0.0)
        + 0.3 * senior
        + 0.06 * (last_login - 12)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)
    flip = rng.random(n) < cfg["label_noise"]
    churn = np.where(flip, 1 - churn, churn)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "contract_type": contract,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "num_products": products,
            "internet_service": internet,
            "senior_citizen": senior,
        }
    )
    if cfg["extra_feature"]:
        df["last_login_days"] = last_login
    df["churned"] = churn

    if cfg["dupes"]:
        dupes = df.sample(cfg["dupes"], random_state=cfg["seed"])
        df = pd.concat([df, dupes], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, choices=sorted(VERSIONS))
    parser.add_argument("--output", default="data/raw/churn.csv")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.version)
    # lineterminator fixed so bytes (and DVC md5) are identical on Windows and Linux
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows x {df.shape[1]} cols -> {out} (dataset {args.version})")


if __name__ == "__main__":
    main()
