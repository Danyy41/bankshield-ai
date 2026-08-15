"""Step 3: split transactions into train/test sets without leakage.

Splits chronologically (see bankshield.features.time_based_split for why)
and saves both halves to data/processed/ for the training scripts to use.

Usage:
    python scripts/03_split_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bankshield import config
from bankshield.features import time_based_split


def main() -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.TRANSACTIONS_CSV, parse_dates=["timestamp"])
    train_df, test_df = time_based_split(df)

    train_df.to_csv(config.TRAIN_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print(f"Train: {len(train_df):,} rows, {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}, "
          f"fraud rate {train_df['is_fraud'].mean():.3%}")
    print(f"Test:  {len(test_df):,} rows, {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}, "
          f"fraud rate {test_df['is_fraud'].mean():.3%}")

    overlap = set(train_df["customer_id"]) & set(test_df["customer_id"])
    print(f"Customers appearing in both splits: {len(overlap):,} "
          f"(expected -- a customer's earlier transactions can train while their later ones test; "
          f"no single transaction appears in both splits)")


if __name__ == "__main__":
    main()
