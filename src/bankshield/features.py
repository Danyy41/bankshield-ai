"""Feature engineering boundary: train/test splitting and preprocessing.

Splitting strategy
-------------------
Transactions are split **by time**, not by a random shuffle: the earliest
`1 - TEST_SIZE_FRACTION` of transactions become the training set, the most
recent `TEST_SIZE_FRACTION` become the test set. Two reasons:

1. Random shuffling would put a customer's transaction from January in
   train and one from February in test, letting the model implicitly
   "see the future" via the customer's later behaviour when predicting
   an earlier date -- and vice versa. In production, a fraud model only
   ever predicts on transactions it has never seen, all of which happen
   *after* the data it was trained on.
2. It gives an honest read on how the model will perform once deployed,
   including any behavioural drift over the simulated period.

Preprocessing
--------------
A single `ColumnTransformer` scales numeric features, one-hot encodes
low-cardinality categoricals, and passes boolean risk flags through as-is.
It is fit only on the training split and reused (never refit) on the test
split, and is bundled into the same `Pipeline` as the classifier so the
saved model artifact is self-contained.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bankshield import config


def time_based_split(
    df: pd.DataFrame, test_size_fraction: float = config.TEST_SIZE_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions chronologically: earliest rows train, latest rows test."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * (1 - test_size_fraction))
    cutoff_time = df_sorted.loc[cutoff_idx, "timestamp"]
    train_df = df_sorted[df_sorted["timestamp"] < cutoff_time].reset_index(drop=True)
    test_df = df_sorted[df_sorted["timestamp"] >= cutoff_time].reset_index(drop=True)
    return train_df, test_df


def get_X_y(
    df: pd.DataFrame, feature_columns: list[str] = config.FEATURE_COLUMNS
) -> tuple[pd.DataFrame, pd.Series]:
    """Select model-input columns and target from a transactions DataFrame.

    Defaults to Phase 1's transaction-only feature set; pass
    `config.FEATURE_COLUMNS_WITH_CYBER` to include Phase 2's cyber
    features instead.
    """
    X = df[feature_columns].copy()
    y = df[config.TARGET_COL].astype(int).copy()
    return X, y


def build_preprocessor(
    numeric_features: list[str] = config.NUMERIC_FEATURES,
    categorical_features: list[str] = config.CATEGORICAL_FEATURES,
    binary_features: list[str] = config.BINARY_FEATURES,
) -> ColumnTransformer:
    """Numeric -> scale, categorical -> one-hot, binary flags -> passthrough.

    Defaults to Phase 1's transaction-only feature groups; pass
    `config.NUMERIC_FEATURES + config.CYBER_NUMERIC_FEATURES` (and the
    binary equivalent) to build a preprocessor for the transaction+cyber
    feature set instead.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("binary", "passthrough", binary_features),
        ]
    )
