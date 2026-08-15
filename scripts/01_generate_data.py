"""Step 1: generate the synthetic transactions dataset and save it to disk.

Usage:
    python scripts/01_generate_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankshield import config
from bankshield.data_generation import generate_transactions


def main() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating ~{config.N_TRANSACTIONS:,} synthetic transactions "
          f"for {config.N_CUSTOMERS:,} customers over {config.SIMULATION_DAYS} days...")
    df = generate_transactions()

    df.to_csv(config.TRANSACTIONS_CSV, index=False)

    fraud_rate = df["is_fraud"].mean()
    print(f"Saved {len(df):,} rows to {config.TRANSACTIONS_CSV}")
    print(f"Fraud rate: {fraud_rate:.3%} ({df['is_fraud'].sum():,} fraudulent transactions)")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Unique customers: {df['customer_id'].nunique():,}")


if __name__ == "__main__":
    main()
