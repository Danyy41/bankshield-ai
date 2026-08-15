"""Step 6: compare baseline vs XGBoost and write the final Phase 1 report.

Reloads both saved pipelines, re-scores the test set, plots ROC/PR curves
side by side, plots XGBoost feature importances, and writes
reports/model_comparison.md -- including a worked explanation of why raw
accuracy is misleading for this problem.

Usage:
    python scripts/06_compare_models.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bankshield import config, evaluation
from bankshield.features import get_X_y


def plot_feature_importance(xgb_pipeline, out_path) -> None:
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_pipeline.named_steps["classifier"].feature_importances_

    order = importances.argsort()[::-1][:15]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    ax.set_title("XGBoost -- top 15 feature importances")
    ax.set_xlabel("Importance (gain-based)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    test_df = pd.read_csv(config.TEST_CSV, parse_dates=["timestamp"])
    X_test, y_test = get_X_y(test_df)

    baseline = joblib.load(config.BASELINE_MODEL_PATH)
    xgb = joblib.load(config.XGBOOST_MODEL_PATH)

    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    evaluation.plot_pr_roc_curves(
        {
            "Logistic Regression": (y_test, baseline_proba),
            "XGBoost": (y_test, xgb_proba),
        },
        config.FIGURES_DIR / "roc_pr_comparison.png",
    )
    plot_feature_importance(xgb, config.FIGURES_DIR / "xgboost_feature_importance.png")

    with open(config.METRICS_DIR / "baseline_metrics.json") as f:
        baseline_metrics = json.load(f)
    with open(config.METRICS_DIR / "xgboost_metrics.json") as f:
        xgb_metrics = json.load(f)

    prevalence = baseline_metrics["fraud_prevalence"]
    naive_acc = baseline_metrics["naive_always_legit_accuracy"]

    def row(name, m):
        return (f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")

    lines = [
        "# BankShield AI -- Phase 1 Model Comparison",
        "",
        f"Test set: {baseline_metrics['n_test_samples']:,} transactions, "
        f"{prevalence:.3%} fraud prevalence.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|---|",
        row("Logistic Regression (baseline)", baseline_metrics),
        row("XGBoost", xgb_metrics),
        "",
        "## Why accuracy alone is misleading here",
        "",
        f"Fraud is rare: only {prevalence:.3%} of test transactions are fraudulent. "
        f"A model that predicts \"legitimate\" for every single transaction -- doing "
        f"zero fraud detection whatsoever -- would score **{naive_acc:.2%} accuracy**, "
        "which sounds excellent in isolation and would beat many real models on that "
        "metric alone. Accuracy weighs every prediction equally, so with a ~1.5% "
        "positive rate it is almost entirely determined by how well the model handles "
        "the 98.5% majority class, telling you next to nothing about whether it ever "
        "catches fraud.",
        "",
        "This is why this project reports precision, recall, F1, ROC-AUC, and PR-AUC "
        "for every model instead of leading with accuracy:",
        "",
        "- **Recall** answers \"of the actual fraud, how much did we catch?\" -- the "
        "metric that matters most to a bank trying to limit losses.",
        "- **Precision** answers \"of the transactions we flagged, how many were "
        "actually fraud?\" -- controls the cost of false alarms (blocked legitimate "
        "customers, manual review workload).",
        "- **F1** balances the two when there's no single dominant business cost.",
        "- **ROC-AUC** measures ranking quality across all thresholds, but can look "
        "artificially strong under heavy class imbalance because the false positive "
        "*rate* stays small even when false positives significantly outnumber true "
        "positives in absolute terms.",
        "- **PR-AUC** is the more honest summary metric for this problem: since it's "
        "computed from precision, it directly reflects how bad the imbalance-driven "
        "false-alarm problem is, and a random classifier's PR-AUC baseline equals the "
        f"prevalence itself (~{prevalence:.1%} here, not 0.5).",
        "",
        "## Confusion matrices",
        "",
        "See `reports/figures/baseline_confusion_matrix.png` and "
        "`reports/figures/xgboost_confusion_matrix.png`.",
        "",
        "## Figures",
        "",
        "- `reports/figures/roc_pr_comparison.png` -- ROC and PR curves, both models",
        "- `reports/figures/xgboost_feature_importance.png` -- which signals XGBoost relies on most",
        "",
    ]

    report_path = config.REPORTS_DIR / "model_comparison.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved comparison report to {report_path}")
    print(f"Saved figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
