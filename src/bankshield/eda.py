"""Exploratory data analysis helpers for the transactions dataset.

Produces a handful of figures and a markdown summary rather than a
notebook, so it can be re-run headlessly as part of the pipeline and
diffed cleanly in version control.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to save PNGs

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def class_balance_summary(df: pd.DataFrame) -> dict:
    counts = df["is_fraud"].value_counts()
    return {
        "total_transactions": int(len(df)),
        "fraud_count": int(counts.get(1, 0)),
        "legit_count": int(counts.get(0, 0)),
        "fraud_rate_pct": round(float(100 * df["is_fraud"].mean()), 3),
    }


def plot_class_balance(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["is_fraud"].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.2%})", ha="center", va="bottom")
    ax.set_title("Class balance: fraud is rare")
    ax.set_ylabel("Transaction count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_amount_distribution(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name, color in [(0, "Legitimate", "#4C72B0"), (1, "Fraud", "#C44E52")]:
        subset = df.loc[df["is_fraud"] == label, "amount"]
        sns.kdeplot(subset.clip(upper=subset.quantile(0.99)), ax=ax, label=name, color=color, fill=True, alpha=0.3)
    ax.set_title("Transaction amount distribution by class")
    ax.set_xlabel("Amount ($, clipped at 99th percentile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fraud_rate_by_hour(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    rate_by_hour = df.groupby("hour_of_day")["is_fraud"].mean() * 100
    ax.bar(rate_by_hour.index, rate_by_hour.values, color="#C44E52")
    ax.set_title("Fraud rate by hour of day")
    ax.set_xlabel("Hour of day (0-23)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fraud_rate_by_category(df: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    rate_by_cat = (df.groupby("merchant_category")["is_fraud"].mean() * 100).sort_values()
    ax.barh(rate_by_cat.index, rate_by_cat.values, color="#55A868")
    ax.set_title("Fraud rate by merchant category")
    ax.set_xlabel("Fraud rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_risk_factor_lift(df: pd.DataFrame, out_path) -> None:
    factors = {
        "new_device": "New device",
        "new_beneficiary": "New beneficiary",
        "country_mismatch": "Foreign country",
        "is_night": "Night-time (0-6am)",
    }
    base_rate = df["is_fraud"].mean()
    lifts = []
    for col, label in factors.items():
        rate_true = df.loc[df[col], "is_fraud"].mean()
        lifts.append((label, rate_true / base_rate))
    lifts_df = pd.DataFrame(lifts, columns=["factor", "lift"]).sort_values("lift")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(lifts_df["factor"], lifts_df["lift"], color="#8172B2")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Baseline rate")
    ax.set_title("Fraud-rate lift when a risk factor is present")
    ax.set_xlabel("Multiplier vs. overall fraud rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame, numeric_cols: list[str], out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[numeric_cols + ["is_fraud"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation matrix (numeric features + target)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
