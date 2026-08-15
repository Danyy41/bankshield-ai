import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from bankshield import config
from bankshield.auth_generation import generate_login_events
from bankshield.cyber_features import NO_RECENT_LOGIN_MINUTES, add_cyber_features
from bankshield.data_generation import generate_transactions
from bankshield.features import build_preprocessor, get_X_y


@pytest.fixture(scope="module")
def small_transactions() -> pd.DataFrame:
    return generate_transactions(n_transactions=4000, n_customers=600, seed=11)


@pytest.fixture(scope="module")
def small_logins(small_transactions) -> pd.DataFrame:
    return generate_login_events(small_transactions, seed=12)


@pytest.fixture(scope="module")
def small_enriched(small_transactions, small_logins) -> pd.DataFrame:
    return add_cyber_features(small_transactions, small_logins)


# --- login event generation ------------------------------------------------


def test_login_events_schema_and_uniqueness(small_logins):
    for col in ["login_id", "session_id", "customer_id", "timestamp", "device_id",
                "login_success", "new_device", "new_location", "session_duration_seconds"]:
        assert col in small_logins.columns
    assert small_logins["login_id"].is_unique
    assert small_logins.isna().sum().sum() == 0


def test_ato_pattern_correlates_with_fraud(small_transactions, small_logins):
    """The whole point of the generator: fraud-linked transactions should be
    preceded by account-takeover-shaped logins far more often than legitimate
    ones (but not deterministically -- both rates should be strictly between
    0 and 1)."""
    logins_sorted = small_logins.sort_values("timestamp").reset_index(drop=True)
    txns_sorted = small_transactions.sort_values("timestamp").reset_index(drop=True)
    logins_sorted["timestamp"] = logins_sorted["timestamp"].astype("datetime64[ns]")
    txns_sorted["timestamp"] = txns_sorted["timestamp"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        txns_sorted,
        logins_sorted[["customer_id", "timestamp", "is_ato_pattern"]].rename(
            columns={"timestamp": "login_timestamp"}
        ),
        by="customer_id",
        left_on="timestamp",
        right_on="login_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=25),
    )

    rate_fraud = merged.loc[merged["is_fraud"] == 1, "is_ato_pattern"].mean()
    rate_legit = merged.loc[merged["is_fraud"] == 0, "is_ato_pattern"].mean()
    assert rate_fraud > rate_legit
    assert 0 < rate_legit < 0.5  # noisy baseline, not zero
    assert 0 < rate_fraud < 1.0  # not every fraud case is ATO-flagged either


# --- cyber feature engineering: leakage safety ------------------------------


def test_cyber_features_no_future_leakage():
    """Hand-crafted, exact-value check that only strictly-prior logins are
    ever used -- the same causality discipline Phase 1 applies to
    transaction_velocity_24h."""
    logins = pd.DataFrame(
        [
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:00:00"),
             "login_success": False, "new_device": False, "new_location": False},
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:05:00"),
             "login_success": True, "new_device": True, "new_location": True},
        ]
    )
    transactions = pd.DataFrame(
        [
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 08:00:00"), "is_fraud": 0},
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:03:00"), "is_fraud": 0},
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:10:00"), "is_fraud": 1},
        ]
    )

    enriched = add_cyber_features(transactions, logins).set_index(transactions.index)

    # Before any login: no history available.
    row0 = enriched.iloc[0]
    assert row0["cyber_failed_logins_1h"] == 0
    assert row0["cyber_login_count_24h"] == 0
    assert row0["cyber_new_device_recent"] == False  # noqa: E712
    assert row0["cyber_minutes_since_last_login"] == NO_RECENT_LOGIN_MINUTES

    # Between the two logins: only the 09:00 (failed, familiar) login is visible --
    # the 09:05 new-device login is in the future and must NOT be counted.
    row1 = enriched.iloc[1]
    assert row1["cyber_new_device_recent"] == False  # noqa: E712
    assert row1["cyber_unusual_country_recent"] == False  # noqa: E712
    assert row1["cyber_failed_logins_1h"] == 1
    assert row1["cyber_minutes_since_last_login"] == pytest.approx(3.0)

    # After both logins: most recent (09:05) was a new device/location, but only
    # one failed attempt is in the trailing 1h, so the composite flag stays False.
    row2 = enriched.iloc[2]
    assert row2["cyber_new_device_recent"] == True  # noqa: E712
    assert row2["cyber_unusual_country_recent"] == True  # noqa: E712
    assert row2["cyber_failed_logins_1h"] == 1
    assert row2["cyber_login_count_24h"] == 2
    assert row2["cyber_recent_suspicious_auth"] == False  # noqa: E712


def test_cyber_features_correlate_with_fraud(small_enriched):
    base = small_enriched.groupby("is_fraud")[config.CYBER_FEATURE_COLUMNS].mean()
    assert base.loc[1, "cyber_failed_logins_1h"] > base.loc[0, "cyber_failed_logins_1h"]
    assert base.loc[1, "cyber_recent_suspicious_auth"] > base.loc[0, "cyber_recent_suspicious_auth"]
    assert base.loc[1, "cyber_new_device_recent"] > base.loc[0, "cyber_new_device_recent"]


def test_cyber_enriched_preserves_transaction_rows_and_columns(small_transactions, small_enriched):
    assert len(small_enriched) == len(small_transactions)
    for col in small_transactions.columns:
        assert col in small_enriched.columns
    for col in config.CYBER_FEATURE_COLUMNS:
        assert col in small_enriched.columns


# --- Phase 1 regression guard ------------------------------------------------


def test_phase1_default_feature_helpers_unchanged(small_transactions):
    """features.py gained optional arguments for Phase 2 -- calling the
    Phase 1 functions with no arguments must still behave exactly as before."""
    X, y = get_X_y(small_transactions)
    assert list(X.columns) == config.FEATURE_COLUMNS

    preprocessor = build_preprocessor()
    transformers_by_name = {name: cols for name, _, cols in preprocessor.transformers}
    assert transformers_by_name["numeric"] == config.NUMERIC_FEATURES
    assert transformers_by_name["categorical"] == config.CATEGORICAL_FEATURES
    assert transformers_by_name["binary"] == config.BINARY_FEATURES
