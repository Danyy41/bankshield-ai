"""Run the entire Phase 1 + Phase 2 pipeline end to end, in order:

Phase 1 (fraud-detection ML baseline):
1. Generate synthetic transactions
2. Exploratory data analysis
3. Chronological train/test split
4. Train + evaluate the baseline (Logistic Regression)
5. Train + evaluate XGBoost
6. Compare models and write the final report

Phase 2 (cyber-authentication telemetry):
7. Generate synthetic login/session events
8. Merge cyber features onto transactions, re-split
9. Train + compare transaction-only vs. transaction+cyber XGBoost

Usage:
    python scripts/run_all.py
"""

import runpy
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_generate_data.py",
    "02_run_eda.py",
    "03_split_data.py",
    "04_train_baseline.py",
    "05_train_xgboost.py",
    "06_compare_models.py",
    "07_generate_login_events.py",
    "08_add_cyber_features.py",
    "09_train_compare_cyber.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 70}\nRunning {step}\n{'=' * 70}")
        runpy.run_path(str(SCRIPTS_DIR / step), run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
