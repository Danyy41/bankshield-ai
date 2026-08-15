"""Tests for the Phase 4 data access layer over Phase 1-3 outputs.

These tests assume `python scripts/run_all.py` has been run so
data/processed, data/raw, and models/ are populated -- same precondition
Phase 1-3's own tests rely on implicitly via the checked-in metrics.
"""

from __future__ import annotations

import pytest

from bankshield.investigation import data_access


@pytest.fixture(scope="module")
def store():
    return data_access.get_store()


def test_get_transaction_returns_all_expected_fields(store):
    txn_id = store.transactions.iloc[0]["transaction_id"]
    row = data_access.get_transaction(txn_id)
    assert row["transaction_id"] == txn_id
    for field in ("customer_id", "amount", "merchant_category", "timestamp", "is_fraud"):
        assert field in row


def test_get_transaction_raises_not_found_for_unknown_id():
    with pytest.raises(data_access.NotFoundError):
        data_access.get_transaction("this-id-does-not-exist")


def test_get_auth_history_only_returns_that_customer(store):
    row = store.transactions.iloc[0]
    events = data_access.get_auth_history(row["customer_id"], limit=50)
    assert events
    assert all(e["customer_id"] == row["customer_id"] for e in events)


def test_get_auth_history_before_cutoff_excludes_later_logins(store):
    """Drill-down history for a specific transaction must never include a
    login that happened at or after that transaction's own timestamp --
    the same causality discipline Phase 2 applies when training features,
    now applied to what the investigation tool is allowed to show."""
    row = store.transactions.iloc[100]
    events = data_access.get_auth_history(row["customer_id"], before=row["timestamp"], limit=50)
    for e in events:
        assert e["timestamp"] < row["timestamp"]


def test_get_auth_history_respects_limit(store):
    # Pick a customer with several login events, if one exists in this slice.
    customer_id = store.logins["customer_id"].value_counts().idxmax()
    events = data_access.get_auth_history(customer_id, limit=2)
    assert len(events) <= 2


def test_get_graph_neighbors_is_symmetric(store):
    """If A is a neighbor of B via a shared device/IP/beneficiary, B must
    be a neighbor of A -- the underlying relationship is undirected."""
    # Find a customer known to have at least one neighbor.
    candidate = None
    for customer_id in store.transactions["customer_id"].unique()[:200]:
        neighbors = data_access.get_graph_neighbors(customer_id)
        if neighbors:
            candidate = (customer_id, neighbors[0]["customer_id"])
            break
    assert candidate is not None, "expected at least one customer with a graph neighbor in the first 200"

    cust_a, cust_b = candidate
    b_neighbors = {n["customer_id"] for n in data_access.get_graph_neighbors(cust_b)}
    assert cust_a in b_neighbors


def test_get_risk_score_returns_valid_probability(store):
    txn_id = store.transactions.iloc[0]["transaction_id"]
    result = data_access.get_risk_score(txn_id)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_tier"] in ("tier_1", "tier_2", "tier_3")
    assert len(result["top_contributing_features"]) > 0


def test_get_risk_score_tier_matches_thresholds(store):
    from bankshield import config

    txn_id = store.transactions.iloc[0]["transaction_id"]
    result = data_access.get_risk_score(txn_id)
    score = result["risk_score"]
    if score >= config.RISK_TIER_1_THRESHOLD:
        assert result["risk_tier"] == "tier_1"
    elif score >= config.RISK_TIER_2_THRESHOLD:
        assert result["risk_tier"] == "tier_2"
    else:
        assert result["risk_tier"] == "tier_3"


def test_get_risk_score_high_risk_transaction_cites_plausible_drivers(store):
    """A transaction flagged tier_1 with elevated cyber/graph signal columns
    should surface at least one cyber_* or graph_* feature among its top
    contributors -- otherwise the explanation wouldn't actually be
    explaining the signal driving the alert."""
    fraud_with_signal = store.transactions[
        (store.transactions["is_fraud"] == 1) & (store.transactions["cyber_recent_suspicious_auth"])
    ]
    if fraud_with_signal.empty:
        pytest.skip("no fraud transaction with cyber_recent_suspicious_auth in this dataset")
    txn_id = fraud_with_signal.iloc[0]["transaction_id"]
    result = data_access.get_risk_score(txn_id)
    top_feature_names = {f["feature"] for f in result["top_contributing_features"]}
    assert any(name.startswith("cyber_") or name.startswith("graph_") for name in top_feature_names)


# --- sample_transactions (backs GET /demo/sample-transactions) -------------


def test_sample_transactions_returns_requested_count():
    samples = data_access.sample_transactions(n=5)
    assert len(samples) == 5


def test_sample_transactions_ids_are_all_valid():
    for row in data_access.sample_transactions(n=5):
        data_access.get_transaction(row["transaction_id"])  # raises NotFoundError if invalid


def test_sample_transactions_scores_match_get_risk_score():
    for row in data_access.sample_transactions(n=5):
        risk = data_access.get_risk_score(row["transaction_id"])
        assert row["risk_score"] == risk["risk_score"]
        assert row["risk_tier"] == risk["risk_tier"]


def test_sample_transactions_shape_is_demo_friendly():
    for row in data_access.sample_transactions(n=5):
        assert set(row.keys()) == {"transaction_id", "risk_score", "risk_tier"}


def test_sample_transactions_is_deterministic():
    assert data_access.sample_transactions(n=5) == data_access.sample_transactions(n=5)


def test_sample_transactions_respects_smaller_n():
    samples = data_access.sample_transactions(n=2)
    assert len(samples) == 2
