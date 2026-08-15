# BankShield AI -- Phase 1 Model Comparison

Test set: 10,000 transactions, 1.380% fraud prevalence.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.8282 | 0.0552 | 0.7101 | 0.1024 | 0.8205 | 0.1360 |
| XGBoost | 0.9791 | 0.2290 | 0.2174 | 0.2230 | 0.8260 | 0.1549 |

## Why accuracy alone is misleading here

Fraud is rare: only 1.380% of test transactions are fraudulent. A model that predicts "legitimate" for every single transaction -- doing zero fraud detection whatsoever -- would score **98.62% accuracy**, which sounds excellent in isolation and would beat many real models on that metric alone. Accuracy weighs every prediction equally, so with a ~1.5% positive rate it is almost entirely determined by how well the model handles the 98.5% majority class, telling you next to nothing about whether it ever catches fraud.

This is why this project reports precision, recall, F1, ROC-AUC, and PR-AUC for every model instead of leading with accuracy:

- **Recall** answers "of the actual fraud, how much did we catch?" -- the metric that matters most to a bank trying to limit losses.
- **Precision** answers "of the transactions we flagged, how many were actually fraud?" -- controls the cost of false alarms (blocked legitimate customers, manual review workload).
- **F1** balances the two when there's no single dominant business cost.
- **ROC-AUC** measures ranking quality across all thresholds, but can look artificially strong under heavy class imbalance because the false positive *rate* stays small even when false positives significantly outnumber true positives in absolute terms.
- **PR-AUC** is the more honest summary metric for this problem: since it's computed from precision, it directly reflects how bad the imbalance-driven false-alarm problem is, and a random classifier's PR-AUC baseline equals the prevalence itself (~1.4% here, not 0.5).

## Confusion matrices

See `reports/figures/baseline_confusion_matrix.png` and `reports/figures/xgboost_confusion_matrix.png`.

## Figures

- `reports/figures/roc_pr_comparison.png` -- ROC and PR curves, both models
- `reports/figures/xgboost_feature_importance.png` -- which signals XGBoost relies on most
