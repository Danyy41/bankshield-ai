"""Step 8 (Phase 2): merge cyber features onto transactions, then re-split.

Applies the exact same `time_based_split` used in Phase 1 to the
cyber-enriched dataset, so train_with_cyber.csv / test_with_cyber.csv
contain the identical rows (by transaction_id) as Phase 1's train.csv /
test.csv -- only with cyber_* columns appended. That's what makes the
transaction-only vs. transaction+cyber comparison in step 9 apples to
apples.

Usage:
    python scripts/08_add_cyber_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bankshield import config
from bankshield.cyber_features import add_cyber_features
from bankshield.features import time_based_split


def main() -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    txns = pd.read_csv(config.TRANSACTIONS_CSV, parse_dates=["timestamp"])
    logins = pd.read_csv(config.LOGIN_EVENTS_CSV, parse_dates=["timestamp"])
    print(f"Loaded {len(txns):,} transactions and {len(logins):,} login events")

    enriched = add_cyber_features(txns, logins)
    enriched.to_csv(config.TRANSACTIONS_WITH_CYBER_CSV, index=False)
    print(f"Saved cyber-enriched transactions to {config.TRANSACTIONS_WITH_CYBER_CSV}")

    train_df, test_df = time_based_split(enriched)
    train_df.to_csv(config.TRAIN_WITH_CYBER_CSV, index=False)
    test_df.to_csv(config.TEST_WITH_CYBER_CSV, index=False)
    print(f"Train (+cyber): {len(train_df):,} rows, Test (+cyber): {len(test_df):,} rows")

    # Sanity check: this split must land on the exact same rows as Phase 1's.
    phase1_train_ids = set(pd.read_csv(config.TRAIN_CSV, usecols=["transaction_id"])["transaction_id"])
    phase1_test_ids = set(pd.read_csv(config.TEST_CSV, usecols=["transaction_id"])["transaction_id"])
    match_train = set(train_df["transaction_id"]) == phase1_train_ids
    match_test = set(test_df["transaction_id"]) == phase1_test_ids
    print(f"Row membership matches Phase 1 split: train={match_train}, test={match_test}")

    print("\nCyber feature averages, fraud vs. legitimate (train set):")
    print(train_df.groupby("is_fraud")[config.CYBER_FEATURE_COLUMNS].mean().T)


if __name__ == "__main__":
    main()
