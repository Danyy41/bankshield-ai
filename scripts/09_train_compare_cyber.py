"""Step 9 (Phase 2): does cyber telemetry improve fraud detection?

Trains two XGBoost models on the identical train/test rows (see step 8):
one using only Phase 1's transaction features, one using transaction +
cyber features. Both use the same hyperparameters and scale_pos_weight
so the comparison isolates the effect of the extra features.

Usage:
    python scripts/09_train_compare_cyber.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bankshield import config, evaluation
from bankshield.features import build_preprocessor, get_X_y
from bankshield.modeling import build_xgboost_pipeline, compute_scale_pos_weight


def plot_feature_importance(pipeline, out_path, title) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    order = importances.argsort()[::-1][:15]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C44E52" if fn.startswith("binary__cyber") or fn.startswith("numeric__cyber")
              else "#4C72B0" for fn in [feature_names[i] for i in order]]
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color=colors[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance (gain-based)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.TRAIN_WITH_CYBER_CSV, parse_dates=["timestamp"])
    test_df = pd.read_csv(config.TEST_WITH_CYBER_CSV, parse_dates=["timestamp"])

    # --- Model A: transaction-only (same feature set as Phase 1) -----------
    X_train_a, y_train = get_X_y(train_df, config.FEATURE_COLUMNS)
    X_test_a, y_test = get_X_y(test_df, config.FEATURE_COLUMNS)
    scale_pos_weight = compute_scale_pos_weight(y_train) ** 0.5

    print(f"Training transaction-only XGBoost (scale_pos_weight={scale_pos_weight:.1f})...")
    pipeline_a = build_xgboost_pipeline(scale_pos_weight)
    pipeline_a.fit(X_train_a, y_train)
    metrics_a = evaluation.evaluate_predictions(
        y_test, pipeline_a.predict(X_test_a), pipeline_a.predict_proba(X_test_a)[:, 1]
    )
    evaluation.print_report("Transaction-only (re-trained, same rows as Phase 1)", metrics_a)

    # --- Model B: transaction + cyber ---------------------------------------
    X_train_b, _ = get_X_y(train_df, config.FEATURE_COLUMNS_WITH_CYBER)
    X_test_b, _ = get_X_y(test_df, config.FEATURE_COLUMNS_WITH_CYBER)
    preprocessor_b = build_preprocessor(
        numeric_features=config.NUMERIC_FEATURES + config.CYBER_NUMERIC_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
        binary_features=config.BINARY_FEATURES + config.CYBER_BINARY_FEATURES,
    )

    print("\nTraining transaction+cyber XGBoost...")
    pipeline_b = build_xgboost_pipeline(scale_pos_weight, preprocessor=preprocessor_b)
    pipeline_b.fit(X_train_b, y_train)
    metrics_b = evaluation.evaluate_predictions(
        y_test, pipeline_b.predict(X_test_b), pipeline_b.predict_proba(X_test_b)[:, 1]
    )
    evaluation.print_report("Transaction + cyber telemetry", metrics_b)

    evaluation.save_metrics(metrics_a, config.METRICS_DIR / "xgboost_transaction_only_rerun_metrics.json")
    evaluation.save_metrics(metrics_b, config.METRICS_DIR / "xgboost_with_cyber_metrics.json")
    joblib.dump(pipeline_b, config.XGBOOST_WITH_CYBER_MODEL_PATH)

    evaluation.plot_pr_roc_curves(
        {"Transaction-only": (y_test, pipeline_a.predict_proba(X_test_a)[:, 1]),
         "Transaction + cyber": (y_test, pipeline_b.predict_proba(X_test_b)[:, 1])},
        config.FIGURES_DIR / "cyber_roc_pr_comparison.png",
    )
    plot_feature_importance(
        pipeline_b, config.FIGURES_DIR / "xgboost_with_cyber_feature_importance.png",
        "Transaction + cyber XGBoost -- top 15 feature importances (red = cyber)",
    )

    # --- Report --------------------------------------------------------------
    def row(name, m):
        return (f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")

    pr_auc_delta = metrics_b["pr_auc"] - metrics_a["pr_auc"]
    roc_auc_delta = metrics_b["roc_auc"] - metrics_a["roc_auc"]
    verdict = (
        "Cyber telemetry **improves** fraud detection on this data: both PR-AUC and "
        "ROC-AUC increase when login features are added."
        if pr_auc_delta > 0 and roc_auc_delta > 0
        else "Cyber telemetry does **not** clearly improve fraud detection on this "
        "data at these thresholds -- see the numbers below."
    )

    lines = [
        "# BankShield AI -- Phase 2: Does Cyber Telemetry Improve Fraud Detection?",
        "",
        f"Test set: {metrics_a['n_test_samples']:,} transactions, "
        f"{metrics_a['fraud_prevalence']:.3%} fraud prevalence. Both models trained on "
        "the identical rows, with the identical XGBoost hyperparameters and "
        "scale_pos_weight -- the only difference is whether cyber_* features are "
        "available to the model.",
        "",
        "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|---|",
        row("Transaction-only", metrics_a),
        row("Transaction + cyber", metrics_b),
        "",
        f"PR-AUC change: {pr_auc_delta:+.4f}. ROC-AUC change: {roc_auc_delta:+.4f}.",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Why this makes sense",
        "",
        "Login events are generated *from* the already-fixed Phase 1 fraud labels: "
        "most fraudulent transactions (75%) are preceded by an account-takeover-shaped "
        "login burst (several failed attempts, then a success from a new device/country) "
        "shortly before the transaction, while legitimate transactions almost always show "
        "an ordinary single login, with a small (3%) rate of benign lookalike bursts. The "
        "cyber features (`cyber_failed_logins_1h`, `cyber_new_device_recent`, "
        "`cyber_unusual_country_recent`, `cyber_recent_suspicious_auth`, ...) are causal "
        "aggregates of that history computed strictly before each transaction's own "
        "timestamp, so this is a legitimate, leakage-free signal -- not the model "
        "peeking at its own label.",
        "",
        "## Figures",
        "",
        "- `reports/figures/cyber_roc_pr_comparison.png` -- ROC/PR curves, both models",
        "- `reports/figures/xgboost_with_cyber_feature_importance.png` -- feature "
        "importances for the transaction+cyber model (cyber features highlighted)",
        "",
    ]
    report_path = config.REPORTS_DIR / "phase2_comparison.md"
    report_path.write_text("\n".join(lines))
    print(f"\nSaved comparison report to {report_path}")


if __name__ == "__main__":
    main()
