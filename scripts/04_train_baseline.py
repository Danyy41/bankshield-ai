"""Step 4: train and evaluate the baseline logistic regression model.

Usage:
    python scripts/04_train_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from bankshield import config, evaluation
from bankshield.features import get_X_y
from bankshield.modeling import build_baseline_pipeline


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.TRAIN_CSV, parse_dates=["timestamp"])
    test_df = pd.read_csv(config.TEST_CSV, parse_dates=["timestamp"])
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    print("Training baseline Logistic Regression pipeline...")
    pipeline = build_baseline_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = evaluation.evaluate_predictions(y_test, y_pred, y_proba)
    evaluation.print_report("Baseline: Logistic Regression", metrics)

    evaluation.save_metrics(metrics, config.METRICS_DIR / "baseline_metrics.json")
    evaluation.plot_confusion_matrix(
        metrics, config.FIGURES_DIR / "baseline_confusion_matrix.png",
        "Baseline (Logistic Regression) -- Confusion Matrix",
    )

    joblib.dump(pipeline, config.BASELINE_MODEL_PATH)
    print(f"\nSaved pipeline to {config.BASELINE_MODEL_PATH}")
    print(f"Saved metrics to {config.METRICS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
