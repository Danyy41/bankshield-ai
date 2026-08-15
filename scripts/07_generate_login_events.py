"""Step 7 (Phase 2): generate synthetic login/session events.

Reads the existing transactions.csv (Phase 1's fixed is_fraud labels)
and generates login history around it -- does not modify Phase 1's data.

Usage:
    python scripts/07_generate_login_events.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bankshield import config
from bankshield.auth_generation import generate_login_events


def main() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    txns = pd.read_csv(config.TRANSACTIONS_CSV, parse_dates=["timestamp"])
    print(f"Loaded {len(txns):,} transactions from {config.TRANSACTIONS_CSV}")

    logins = generate_login_events(txns)
    logins.to_csv(config.LOGIN_EVENTS_CSV, index=False)

    print(f"Saved {len(logins):,} login events to {config.LOGIN_EVENTS_CSV}")
    print(f"Login success rate: {logins['login_success'].mean():.2%}")
    print(f"Account-takeover-pattern sessions: {logins['is_ato_pattern'].mean():.2%} of login rows")
    print(f"New device rate: {logins['new_device'].mean():.2%}")
    print(f"Unusual country rate: {logins['new_location'].mean():.2%}")


if __name__ == "__main__":
    main()
