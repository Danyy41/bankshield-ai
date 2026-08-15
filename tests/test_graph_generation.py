import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from bankshield import config
from bankshield.auth_generation import generate_login_events
from bankshield.cyber_features import add_cyber_features
from bankshield.data_generation import generate_transactions
from bankshield.features import get_X_y
from bankshield.graph_features import add_graph_features
from bankshield.graph_generation import _reconstruct_beneficiary_ids, generate_graph_relationships


@pytest.fixture(scope="module")
def small_transactions() -> pd.DataFrame:
    return generate_transactions(n_transactions=5000, n_customers=750, seed=21)


@pytest.fixture(scope="module")
def small_graph(small_transactions) -> pd.DataFrame:
    return generate_graph_relationships(small_transactions, seed=22)


@pytest.fixture(scope="module")
def small_with_cyber(small_transactions) -> pd.DataFrame:
    logins = generate_login_events(small_transactions, seed=23)
    return add_cyber_features(small_transactions, logins)


@pytest.fixture(scope="module")
def small_enriched(small_with_cyber, small_graph) -> pd.DataFrame:
    return add_graph_features(small_with_cyber, small_graph)


# --- graph relationship generation ------------------------------------------


def test_graph_entities_schema(small_graph):
    for col in ["transaction_id", "customer_id", "beneficiary_id", "graph_device_id",
                "graph_ip_address", "ring_id", "is_ring_member"]:
        assert col in small_graph.columns
    assert small_graph["transaction_id"].is_unique


def test_ring_membership_correlates_with_fraud_but_is_not_deterministic(small_transactions, small_graph):
    merged = small_graph.merge(small_transactions[["transaction_id", "is_fraud"]], on="transaction_id")
    rate_member = merged.loc[merged["is_ring_member"], "is_fraud"].mean()
    rate_non_member = merged.loc[~merged["is_ring_member"], "is_fraud"].mean()
    assert rate_member > rate_non_member
    assert 0 < rate_member < 1.0  # not every ring member's transaction is fraud
    assert rate_non_member > 0  # rings aren't the only source of fraud either


def test_beneficiary_reconstruction_consistent_with_new_beneficiary_flag():
    """Hand-crafted check that beneficiary IDs respect the existing
    new_beneficiary flag: a fresh ID on 'new', reuse of a prior ID
    (from the same customer only) otherwise."""
    transactions = pd.DataFrame(
        [
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:00"),
             "merchant_category": "wire_transfer", "new_beneficiary": True},
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:10"),
             "merchant_category": "wire_transfer", "new_beneficiary": False},
            {"customer_id": "C2", "timestamp": pd.Timestamp("2026-01-01 09:05"),
             "merchant_category": "cash_withdrawal", "new_beneficiary": True},
            {"customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:20"),
             "merchant_category": "grocery", "new_beneficiary": False},
        ]
    )
    rng = np.random.default_rng(0)
    beneficiary_id = _reconstruct_beneficiary_ids(transactions, rng)

    assert pd.notna(beneficiary_id.iloc[0])  # C1's first wire transfer: fresh ID
    assert beneficiary_id.iloc[1] == beneficiary_id.iloc[0]  # C1's reuse: same ID
    assert pd.notna(beneficiary_id.iloc[2])
    assert beneficiary_id.iloc[2] != beneficiary_id.iloc[0]  # different customer, different ID
    assert pd.isna(beneficiary_id.iloc[3])  # not a beneficiary-bearing category at all


# --- graph feature engineering: leakage safety ------------------------------


def test_graph_features_no_future_leakage():
    """A customer's graph features must never reflect edges or fraud
    outcomes that are only established by transactions later in time --
    even when a shared device will *eventually* connect them to someone
    else, a transaction before that connection exists must not see it."""
    transactions = pd.DataFrame(
        [
            {"transaction_id": "T1", "customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:00"), "is_fraud": 0},
            {"transaction_id": "T1b", "customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:04"), "is_fraud": 0},
            {"transaction_id": "T2", "customer_id": "C2", "timestamp": pd.Timestamp("2026-01-01 09:05"), "is_fraud": 1},
            {"transaction_id": "T3", "customer_id": "C1", "timestamp": pd.Timestamp("2026-01-01 09:10"), "is_fraud": 0},
        ]
    )
    graph = pd.DataFrame(
        [
            {"transaction_id": "T1", "beneficiary_id": None, "graph_device_id": "dev_shared", "graph_ip_address": "ip1"},
            {"transaction_id": "T1b", "beneficiary_id": None, "graph_device_id": "dev_shared", "graph_ip_address": "ip1"},
            {"transaction_id": "T2", "beneficiary_id": None, "graph_device_id": "dev_shared", "graph_ip_address": "ip2"},
            {"transaction_id": "T3", "beneficiary_id": None, "graph_device_id": "dev_shared", "graph_ip_address": "ip1"},
        ]
    )

    enriched = add_graph_features(transactions, graph).set_index("transaction_id")

    # T1: first ever use of the device -- no one to share it with yet.
    assert enriched.loc["T1", "graph_shared_device_count"] == 0
    assert enriched.loc["T1", "graph_suspicious_neighbor_count"] == 0

    # T1b: only C1 has used this device so far (C2 hasn't happened yet) --
    # must NOT already reflect the fact that C2 will use it one minute later.
    assert enriched.loc["T1b", "graph_shared_device_count"] == 0
    assert enriched.loc["T1b", "graph_suspicious_neighbor_count"] == 0

    # T2: C2 shares the device with C1 (who used it twice before). C1 hasn't
    # committed fraud, so C2 shouldn't show a suspicious neighbor yet.
    assert enriched.loc["T2", "graph_shared_device_count"] == 1
    assert enriched.loc["T2", "graph_suspicious_neighbor_count"] == 0

    # T3: now happens AFTER C2's fraudulent T2, and C1-C2 are connected --
    # C1 SHOULD see C2 as a suspicious neighbor (this is genuine prior
    # information, not leakage).
    assert enriched.loc["T3", "graph_shared_device_count"] == 1
    assert enriched.loc["T3", "graph_suspicious_neighbor_count"] == 1
    assert enriched.loc["T3", "graph_account_network_risk"] == pytest.approx(1.0)


def test_graph_features_correlate_with_fraud(small_enriched):
    base = small_enriched.groupby("is_fraud")[config.GRAPH_FEATURE_COLUMNS].mean()
    assert base.loc[1, "graph_shared_device_count"] > base.loc[0, "graph_shared_device_count"]
    assert base.loc[1, "graph_suspicious_neighbor_count"] > base.loc[0, "graph_suspicious_neighbor_count"]


def test_graph_enriched_preserves_transaction_rows_and_columns(small_transactions, small_enriched):
    assert len(small_enriched) == len(small_transactions)
    for col in config.GRAPH_FEATURE_COLUMNS:
        assert col in small_enriched.columns
    # The raw graph-entity columns are internal to feature engineering and
    # should not leak into the modeling dataframe.
    for col in ["beneficiary_id", "graph_device_id", "graph_ip_address"]:
        assert col not in small_enriched.columns


def test_feature_columns_with_graph_selectable(small_enriched):
    X, y = get_X_y(small_enriched, config.FEATURE_COLUMNS_WITH_GRAPH)
    assert list(X.columns) == config.FEATURE_COLUMNS_WITH_GRAPH
    assert set(y.unique()) <= {0, 1}
