"""Step 5: train and evaluate the XGBoost model.

Usage:
    python scripts/05_train_xgboost.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from bankshield import config, evaluation
from bankshield.features import get_X_y
from bankshield.modeling import build_xgboost_pipeline, compute_scale_pos_weight


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.TRAIN_CSV, parse_dates=["timestamp"])
    test_df = pd.read_csv(config.TEST_CSV, parse_dates=["timestamp"])
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    # The exact class-count ratio (~64:1) over-corrects in practice and floods
    # predictions with false positives; sqrt-dampening it is a common,
    # simple heuristic that keeps recall high without destroying precision.
    raw_ratio = compute_scale_pos_weight(y_train)
    scale_pos_weight = raw_ratio**0.5
    print(f"Training XGBoost pipeline (class ratio={raw_ratio:.1f}, "
          f"scale_pos_weight={scale_pos_weight:.1f})...")
    pipeline = build_xgboost_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = evaluation.evaluate_predictions(y_test, y_pred, y_proba)
    evaluation.print_report("XGBoost", metrics)

    evaluation.save_metrics(metrics, config.METRICS_DIR / "xgboost_metrics.json")
    evaluation.plot_confusion_matrix(
        metrics, config.FIGURES_DIR / "xgboost_confusion_matrix.png",
        "XGBoost -- Confusion Matrix",
    )

    joblib.dump(pipeline, config.XGBOOST_MODEL_PATH)
    print(f"\nSaved pipeline to {config.XGBOOST_MODEL_PATH}")
    print(f"Saved metrics to {config.METRICS_DIR / 'xgboost_metrics.json'}")


if __name__ == "__main__":
    main()
