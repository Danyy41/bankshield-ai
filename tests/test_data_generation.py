import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from bankshield.data_generation import generate_transactions
from bankshield.features import time_based_split


@pytest.fixture(scope="module")
def small_dataset() -> pd.DataFrame:
    return generate_transactions(n_transactions=4000, n_customers=600, seed=7)


def test_row_and_column_shape(small_dataset):
    assert len(small_dataset) == 4000
    for col in ["transaction_id", "customer_id", "timestamp", "amount", "is_fraud"]:
        assert col in small_dataset.columns


def test_transaction_ids_unique(small_dataset):
    assert small_dataset["transaction_id"].is_unique


def test_fraud_is_rare_but_present(small_dataset):
    rate = small_dataset["is_fraud"].mean()
    assert 0 < rate < 0.05, f"fraud rate {rate} outside expected rare-event range"
    assert small_dataset["is_fraud"].sum() > 0


def test_no_missing_values(small_dataset):
    assert small_dataset.isna().sum().sum() == 0


def test_amounts_positive(small_dataset):
    assert (small_dataset["amount"] > 0).all()


def test_risk_factors_correlate_with_fraud(small_dataset):
    """Sanity check that the generator actually produced signal, not noise."""
    base_rate = small_dataset["is_fraud"].mean()
    for col in ["new_device", "new_beneficiary", "country_mismatch"]:
        rate_when_true = small_dataset.loc[small_dataset[col], "is_fraud"].mean()
        assert rate_when_true > base_rate, f"{col} should lift fraud rate above baseline"


def test_fraud_is_not_trivially_separable(small_dataset):
    """No single risk factor should perfectly separate the classes -- otherwise
    the classification task would be trivial rather than realistic."""
    for col in ["new_device", "new_beneficiary", "country_mismatch", "is_night"]:
        rate_when_true = small_dataset.loc[small_dataset[col], "is_fraud"].mean()
        assert rate_when_true < 0.95, f"{col} alone should not near-perfectly predict fraud"


def test_time_based_split_has_no_row_overlap(small_dataset):
    train_df, test_df = time_based_split(small_dataset)
    assert set(train_df["transaction_id"]).isdisjoint(set(test_df["transaction_id"]))
    assert len(train_df) + len(test_df) == len(small_dataset)


def test_time_based_split_is_chronological(small_dataset):
    train_df, test_df = time_based_split(small_dataset)
    assert train_df["timestamp"].max() <= test_df["timestamp"].min()
