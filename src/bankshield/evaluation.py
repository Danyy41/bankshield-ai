"""Evaluation metrics and plots for imbalanced fraud classification.

Why not just accuracy?
------------------------
With fraud at ~1.5% prevalence, a classifier that predicts "not fraud"
for every single transaction scores ~98.5% accuracy while catching zero
fraud -- the one outcome the model exists to prevent. Accuracy is
dominated by the majority class and tells you almost nothing about
performance on the minority class you actually care about. This module
always reports precision, recall, F1, ROC-AUC, and PR-AUC alongside
accuracy so that trade-off is visible rather than hidden.

PR-AUC in particular is the more informative summary metric here: ROC-AUC
can look deceptively strong under heavy class imbalance because the false
positive *rate* stays low even when the absolute number of false alarms
relative to true fraud is large; PR-AUC is sensitive to exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    naive_accuracy = 1 - float(np.mean(y_true))  # "always predict not-fraud" baseline
    return {
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "naive_always_legit_accuracy": naive_accuracy,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fraud_prevalence": float(np.mean(y_true)),
        "n_test_samples": int(len(y_true)),
    }


def print_report(model_name: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}  "
          f"(naive 'always legitimate' baseline would score {metrics['naive_always_legit_accuracy']:.4f} "
          f"-- accuracy alone is misleading here)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"Confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(metrics: dict, out_path: Path, title: str) -> None:
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]:,}", ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted legit", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_roc_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    """results: {model_name: (y_true, y_proba)}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for name, (y_true, y_proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend()

    for name, (y_true, y_proba) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        axes[1].plot(recall, precision, label=f"{name} (AP={average_precision_score(y_true, y_proba):.3f})")
    prevalence = float(np.mean(next(iter(results.values()))[0]))
    axes[1].axhline(prevalence, color="black", linestyle="--", linewidth=1, label="Random (= prevalence)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
