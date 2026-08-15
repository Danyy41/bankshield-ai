"""Central configuration for BankShield AI.

Every script/module pulls paths, column lists, and run parameters from
here so later phases can extend the pipeline (e.g. add new feature
groups, new data sources) without hunting for hard-coded values.

Phase 1 constants (transactions, splits, transaction-only features) are
untouched by Phase 2 -- the cyber-telemetry additions live in their own
section below and are purely additive.
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


# =========================================================================
# Phase 2: cyber-authentication telemetry
# =========================================================================
# Everything below is additive -- it reads Phase 1's transactions.csv (with
# its already-fixed is_fraud labels) and never modifies it or the constants
# above.

LOGIN_EVENTS_CSV = DATA_RAW_DIR / "login_events.csv"
TRANSACTIONS_WITH_CYBER_CSV = DATA_PROCESSED_DIR / "transactions_with_cyber.csv"
TRAIN_WITH_CYBER_CSV = DATA_PROCESSED_DIR / "train_with_cyber.csv"
TEST_WITH_CYBER_CSV = DATA_PROCESSED_DIR / "test_with_cyber.csv"

XGBOOST_WITH_CYBER_MODEL_PATH = MODELS_DIR / "xgboost_with_cyber_pipeline.joblib"

AUTH_RANDOM_SEED = RANDOM_SEED + 1  # separate RNG stream from transaction generation

# Login-generation behaviour. Every transaction gets a "pre-auth" login
# session shortly before it; the probabilities below decide what shape
# that session takes. Baseline "routine" logins (not tied to any
# transaction, e.g. balance checks) are added on top for realism.
ATO_INJECTION_RATE_FRAUD = 0.75  # fraction of fraudulent transactions preceded by an ATO-pattern login
ATO_FALSE_ALARM_RATE = 0.03  # fraction of legitimate transactions that still get a benign ATO-like login (noise)
ROUTINE_LOGINS_PER_CUSTOMER_MEAN = 2.5  # extra standalone logins per customer, unrelated to any transaction

# Causal lookback windows used when aggregating a customer's login history
# onto a transaction (see cyber_features.py) -- only logins strictly before
# the transaction's own timestamp are ever considered.
CYBER_FAILED_LOGIN_WINDOW_HOURS = 1
CYBER_LOGIN_COUNT_WINDOW_HOURS = 24
CYBER_RECENT_LOGIN_LOOKBACK_HOURS = 24  # how far back to look for "most recent login" context

CYBER_NUMERIC_FEATURES = [
    "cyber_failed_logins_1h",
    "cyber_login_count_24h",
    "cyber_minutes_since_last_login",
]

CYBER_BINARY_FEATURES = [
    "cyber_new_device_recent",
    "cyber_unusual_country_recent",
    "cyber_recent_suspicious_auth",
]

CYBER_FEATURE_COLUMNS = CYBER_NUMERIC_FEATURES + CYBER_BINARY_FEATURES

# Transaction-only feature set (Phase 1, unchanged) + cyber telemetry.
FEATURE_COLUMNS_WITH_CYBER = FEATURE_COLUMNS + CYBER_FEATURE_COLUMNS


# =========================================================================
# Phase 3: graph-based financial crime intelligence
# =========================================================================
# Everything below is additive -- it reads Phase 1/2's already-fixed data
# and never modifies it or the constants above.

GRAPH_ENTITIES_CSV = DATA_RAW_DIR / "graph_entities.csv"
TRANSACTIONS_WITH_GRAPH_CSV = DATA_PROCESSED_DIR / "transactions_with_graph.csv"
TRAIN_WITH_GRAPH_CSV = DATA_PROCESSED_DIR / "train_with_graph.csv"
TEST_WITH_GRAPH_CSV = DATA_PROCESSED_DIR / "test_with_graph.csv"

XGBOOST_WITH_GRAPH_MODEL_PATH = MODELS_DIR / "xgboost_with_graph_pipeline.joblib"

GRAPH_RANDOM_SEED = RANDOM_SEED + 2  # separate RNG stream from transaction/login generation

# Mule-ring simulation. Real device/IP/beneficiary collisions are ~never
# organic in this synthetic data (each is independently randomized in
# Phase 1/2), so realistic shared-infrastructure structure has to be
# deliberately injected -- see graph_generation.py.
N_RINGS = 25
RING_SIZE_MIN = 3
RING_SIZE_MAX = 7
RING_FRAUD_MEMBER_BIAS = 0.7  # probability a ring member is drawn from the fraud-customer pool vs. a random customer
RING_TRANSACTION_OVERRIDE_RATE = 0.4  # probability an eligible transaction's device/IP is reassigned to the ring's shared pool
RING_BENEFICIARY_OVERRIDE_RATE = 0.5  # probability a ring member's new-beneficiary transaction points at a shared ring beneficiary

# Causal lookback: graph features are built from a single chronological pass
# over ALL transactions (the graph connects across customers, unlike Phase
# 1/2's per-customer causal loops); see graph_features.py.
GRAPH_NUMERIC_FEATURES = [
    "graph_shared_device_count",
    "graph_shared_ip_count",
    "graph_beneficiary_connectivity",
    "graph_suspicious_neighbor_count",
    "graph_account_network_risk",
]

GRAPH_FEATURE_COLUMNS = GRAPH_NUMERIC_FEATURES

# Transaction + cyber (Phase 1+2, unchanged) + graph telemetry.
FEATURE_COLUMNS_WITH_GRAPH = FEATURE_COLUMNS_WITH_CYBER + GRAPH_FEATURE_COLUMNS
