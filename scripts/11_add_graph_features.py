"""Step 11 (Phase 3): merge graph features onto transactions, then re-split.

Builds on the Phase 2 cyber-enriched dataset (so the resulting file has
transaction + cyber + graph columns together) and re-splits with the
exact same `time_based_split` used in Phase 1/2, so train_with_graph.csv
/ test_with_graph.csv contain the identical rows (by transaction_id) as
the earlier splits.

Usage:
    python scripts/11_add_graph_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bankshield import config
from bankshield.features import time_based_split
from bankshield.graph_features import add_graph_features


def main() -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    txns_with_cyber = pd.read_csv(config.TRANSACTIONS_WITH_CYBER_CSV, parse_dates=["timestamp"])
    graph = pd.read_csv(config.GRAPH_ENTITIES_CSV)
    print(f"Loaded {len(txns_with_cyber):,} cyber-enriched transactions and {len(graph):,} graph-entity rows")

    enriched = add_graph_features(txns_with_cyber, graph)
    enriched.to_csv(config.TRANSACTIONS_WITH_GRAPH_CSV, index=False)
    print(f"Saved transaction+cyber+graph dataset to {config.TRANSACTIONS_WITH_GRAPH_CSV}")

    train_df, test_df = time_based_split(enriched)
    train_df.to_csv(config.TRAIN_WITH_GRAPH_CSV, index=False)
    test_df.to_csv(config.TEST_WITH_GRAPH_CSV, index=False)
    print(f"Train (+graph): {len(train_df):,} rows, Test (+graph): {len(test_df):,} rows")

    # Sanity check: this split must land on the exact same rows as Phase 1/2's.
    phase1_train_ids = set(pd.read_csv(config.TRAIN_CSV, usecols=["transaction_id"])["transaction_id"])
    phase1_test_ids = set(pd.read_csv(config.TEST_CSV, usecols=["transaction_id"])["transaction_id"])
    match_train = set(train_df["transaction_id"]) == phase1_train_ids
    match_test = set(test_df["transaction_id"]) == phase1_test_ids
    print(f"Row membership matches Phase 1 split: train={match_train}, test={match_test}")

    print("\nGraph feature averages, fraud vs. legitimate (train set):")
    print(train_df.groupby("is_fraud")[config.GRAPH_FEATURE_COLUMNS].mean().T)


if __name__ == "__main__":
    main()
