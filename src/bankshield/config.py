"""Central configuration for Phase 1.

Every script/module pulls paths, column lists, and run parameters from
here so later phases can extend the pipeline (e.g. add new feature
groups, new data sources) without hunting for hard-coded values.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

TRANSACTIONS_CSV = DATA_RAW_DIR / "transactions.csv"
TRAIN_CSV = DATA_PROCESSED_DIR / "train.csv"
TEST_CSV = DATA_PROCESSED_DIR / "test.csv"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_logreg_pipeline.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_pipeline.joblib"

# --- Reproducibility -----------------------------------------------------
RANDOM_SEED = 42

# --- Synthetic data generation ------------------------------------------
N_TRANSACTIONS = 50_000
N_CUSTOMERS = 8_000
TARGET_FRAUD_RATE = 0.015  # ~1.5% of transactions, deliberately rare
SIMULATION_DAYS = 120  # transactions span this many days

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "fuel",
    "utilities",
    "online_retail",
    "electronics",
    "travel",
    "healthcare",
    "entertainment",
    "cash_withdrawal",
    "wire_transfer",
    "gambling",
    "crypto_exchange",
]
# Categories that are intrinsically higher-risk / more attractive to fraudsters.
HIGH_RISK_CATEGORIES = {"gambling", "crypto_exchange", "wire_transfer", "cash_withdrawal"}

HOME_COUNTRIES = ["US", "GB", "DE", "FR", "CA", "AU", "NL", "ES"]
FOREIGN_COUNTRIES = HOME_COUNTRIES + ["NG", "RU", "CN", "BR", "IN", "AE", "SG", "ZA"]

# --- Feature groups (used by the preprocessing ColumnTransformer) -------
TARGET_COL = "is_fraud"

NUMERIC_FEATURES = [
    "amount",
    "account_age_days",
    "transaction_velocity_24h",
    "hour_of_day",
    "day_of_week",
    "amount_to_avg_ratio",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "country",
]

BINARY_FEATURES = [
    "new_device",
    "new_beneficiary",
    "country_mismatch",
    "is_night",
]

# Identifier / bookkeeping columns kept in the dataset but excluded from
# model input (would either leak the target or provide no signal, e.g.
# raw IDs and free-form IP strings).
ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "device_id",
    "ip_address",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# --- Train/test split -----------------------------------------------------
# Time-based split: train on the earlier portion, test on the most recent
# transactions. This mirrors real deployment (predict the future from the
# past) and avoids the leakage a random shuffle would cause when the same
# customer appears in both sets with near-identical rolling features.
TEST_SIZE_FRACTION = 0.2
