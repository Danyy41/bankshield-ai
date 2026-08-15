# BankShield AI -- Phase 3: Does Graph Intelligence Improve Fraud Detection Further?

Test set: 10,000 transactions, 1.380% fraud prevalence. Both models trained on the identical rows, with the identical XGBoost hyperparameters and scale_pos_weight -- the only difference is whether graph_* features are available on top of transaction + cyber.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Transaction + cyber | 0.4320 | 0.6449 | 0.5174 | 0.9247 | 0.5508 |
| Transaction + cyber + graph | 0.4279 | 0.6232 | 0.5074 | 0.9373 | 0.5717 |

PR-AUC change: +0.0209. ROC-AUC change: +0.0126.

**Verdict:** Graph intelligence **further improves** fraud detection on this data: both PR-AUC and ROC-AUC increase when graph features are added on top of transaction + cyber. At the default 0.5 threshold, precision/recall/F1 actually dip slightly despite both AUCs improving -- adding features shifts the model's predicted-probability distribution, so the fixed cutoff that suited the transaction+cyber model isn't automatically optimal for transaction+cyber+graph. This is a threshold-calibration artifact, not evidence against the features: a model that ranks fraud better (higher AUCs) but is read through a stale cutoff can still score worse on cutoff-dependent metrics. A deployment would re-tune the threshold before shipping either model.

## Why this makes sense

Mule rings are generated independently of the fraud labels' other drivers: ~25 rings of 3-7 customers each, membership skewed toward (not limited to) customers who already have a fraudulent transaction, sharing a small pool of devices/IPs/beneficiary IDs across a fraction of their transactions (biased toward the fraudulent ones). Ring members show an 11-12% fraud rate versus ~1.3% for everyone else -- real but noisy structure, not a deterministic tell. The graph_* features (`graph_shared_device_count`, `graph_suspicious_neighbor_count`, `graph_account_network_risk`, ...) are built from a single chronological pass that only ever reads graph state accumulated strictly before each transaction's own timestamp, so this is a legitimate, leakage-free signal.

## Figures

- `reports/figures/graph_roc_pr_comparison.png` -- ROC/PR curves, both models
- `reports/figures/xgboost_with_graph_feature_importance.png` -- feature importances for the transaction+cyber+graph model (graph features highlighted)
