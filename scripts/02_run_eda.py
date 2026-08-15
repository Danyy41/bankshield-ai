"""Step 2: exploratory data analysis on the generated transactions.

Saves figures to reports/figures/ and a markdown summary to
reports/eda_summary.md.

Usage:
    python scripts/02_run_eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from bankshield import config, eda


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.TRANSACTIONS_CSV, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} transactions from {config.TRANSACTIONS_CSV}")

    summary = eda.class_balance_summary(df)
    print("Class balance:", summary)

    eda.plot_class_balance(df, config.FIGURES_DIR / "class_balance.png")
    eda.plot_amount_distribution(df, config.FIGURES_DIR / "amount_distribution.png")
    eda.plot_fraud_rate_by_hour(df, config.FIGURES_DIR / "fraud_rate_by_hour.png")
    eda.plot_fraud_rate_by_category(df, config.FIGURES_DIR / "fraud_rate_by_category.png")
    eda.plot_risk_factor_lift(df, config.FIGURES_DIR / "risk_factor_lift.png")
    eda.plot_numeric_correlation(df, config.NUMERIC_FEATURES, config.FIGURES_DIR / "correlation_matrix.png")
    print(f"Saved figures to {config.FIGURES_DIR}")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    lines = [
        "# BankShield AI -- Phase 1 EDA Summary",
        "",
        f"- Total transactions: {summary['total_transactions']:,}",
        f"- Fraudulent transactions: {summary['fraud_count']:,} ({summary['fraud_rate_pct']}%)",
        f"- Legitimate transactions: {summary['legit_count']:,}",
        f"- Missing values: {'none' if missing.empty else missing.to_dict()}",
        "",
        "## Fraud rate by risk factor",
        "",
    ]
    for col in ["new_device", "new_beneficiary", "country_mismatch", "is_night"]:
        rates = df.groupby(col)["is_fraud"].mean() * 100
        lines.append(f"- `{col}`: False={rates.get(False, 0):.2f}%  True={rates.get(True, 0):.2f}%")

    lines += ["", "## Fraud rate by merchant category", ""]
    for cat, rate in (df.groupby("merchant_category")["is_fraud"].mean() * 100).sort_values(ascending=False).items():
        lines.append(f"- `{cat}`: {rate:.2f}%")

    lines += [
        "",
        "## Figures",
        "",
        "- `reports/figures/class_balance.png`",
        "- `reports/figures/amount_distribution.png`",
        "- `reports/figures/fraud_rate_by_hour.png`",
        "- `reports/figures/fraud_rate_by_category.png`",
        "- `reports/figures/risk_factor_lift.png`",
        "- `reports/figures/correlation_matrix.png`",
        "",
    ]

    summary_path = config.REPORTS_DIR / "eda_summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"Saved EDA summary to {summary_path}")


if __name__ == "__main__":
    main()
