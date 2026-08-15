# BankShield AI -- Phase 2: Does Cyber Telemetry Improve Fraud Detection?

Test set: 10,000 transactions, 1.380% fraud prevalence. Both models trained on the identical rows, with the identical XGBoost hyperparameters and scale_pos_weight -- the only difference is whether cyber_* features are available to the model.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Transaction-only | 0.2290 | 0.2174 | 0.2230 | 0.8260 | 0.1549 |
| Transaction + cyber | 0.4320 | 0.6449 | 0.5174 | 0.9247 | 0.5508 |

PR-AUC change: +0.3958. ROC-AUC change: +0.0986.

**Verdict:** Cyber telemetry **improves** fraud detection on this data: both PR-AUC and ROC-AUC increase when login features are added.

## Why this makes sense

Login events are generated *from* the already-fixed Phase 1 fraud labels: most fraudulent transactions (75%) are preceded by an account-takeover-shaped login burst (several failed attempts, then a success from a new device/country) shortly before the transaction, while legitimate transactions almost always show an ordinary single login, with a small (3%) rate of benign lookalike bursts. The cyber features (`cyber_failed_logins_1h`, `cyber_new_device_recent`, `cyber_unusual_country_recent`, `cyber_recent_suspicious_auth`, ...) are causal aggregates of that history computed strictly before each transaction's own timestamp, so this is a legitimate, leakage-free signal -- not the model peeking at its own label.

## Figures

- `reports/figures/cyber_roc_pr_comparison.png` -- ROC/PR curves, both models
- `reports/figures/xgboost_with_cyber_feature_importance.png` -- feature importances for the transaction+cyber model (cyber features highlighted)
