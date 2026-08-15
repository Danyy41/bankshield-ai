"""Model definitions: preprocessing + classifier bundled into one Pipeline.

Bundling the preprocessor and classifier means the artifact saved to disk
is self-contained -- `joblib.load(...).predict(raw_dataframe)` works
without the caller needing to know how features were engineered.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from bankshield import config
from bankshield.features import build_preprocessor


def compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost's scale_pos_weight = (negative count / positive count)."""
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / pos if pos else 1.0


def build_baseline_pipeline() -> Pipeline:
    """Logistic regression: fast, interpretable, standard first baseline
    for a rare-event classification problem."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # fraud is ~1.5% of rows
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    """Gradient-boosted trees: captures non-linear interactions between
    risk factors (e.g. new_device AND night-time AND high_risk_category)
    that a linear model can only approximate."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
