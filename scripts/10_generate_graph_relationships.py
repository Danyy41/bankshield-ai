"""Step 10 (Phase 3): generate the graph-relationship layer.

Reads the existing transactions.csv (Phase 1's fixed is_fraud labels)
and generates reconstructed beneficiary IDs + mule-ring-injected shared
device/IP/beneficiary identifiers -- does not modify Phase 1's data.

Usage:
    python scripts/10_generate_graph_relationships.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bankshield import config
from bankshield.graph_generation import generate_graph_relationships


def main() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    txns = pd.read_csv(config.TRANSACTIONS_CSV, parse_dates=["timestamp"])
    print(f"Loaded {len(txns):,} transactions from {config.TRANSACTIONS_CSV}")

    graph = generate_graph_relationships(txns)
    graph.to_csv(config.GRAPH_ENTITIES_CSV, index=False)

    print(f"Saved {len(graph):,} graph-entity rows to {config.GRAPH_ENTITIES_CSV}")
    print(f"Rings: {graph['ring_id'].nunique()}, ring-member transaction rows: {graph['is_ring_member'].sum():,}")
    print(f"Beneficiary IDs reconstructed for {graph['beneficiary_id'].notna().sum():,} transactions")

    merged = graph.merge(txns[["transaction_id", "is_fraud"]], on="transaction_id")
    fraud_rate_ring = merged.loc[merged["is_ring_member"], "is_fraud"].mean()
    fraud_rate_non_ring = merged.loc[~merged["is_ring_member"], "is_fraud"].mean()
    print(f"Fraud rate: ring members {fraud_rate_ring:.2%} vs. non-members {fraud_rate_non_ring:.2%}")


if __name__ == "__main__":
    main()
