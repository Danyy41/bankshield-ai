"""Data access layer over Phase 1-3 outputs.

This is the one place Phase 4 reads Phase 1/2/3 artifacts (the trained
model, the feature-enriched transaction table, login events, graph
entities). Everything here is read-only against files those phases already
produce -- no Phase 1-3 module is modified, and nothing here re-trains or
re-derives features; it merely looks them up.

Loaded once per process (`get_store()`), not on every request -- these are
multi-hundred-KB CSVs, cheap to hold in memory for a demo service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from bankshield import config

MODEL_VERSION = "xgboost_with_graph_v1"


class NotFoundError(Exception):
    """Raised when a lookup (transaction, customer) has no match."""


@dataclass
class GraphIndex:
    """Non-causal (whole-history) neighbor index for investigation
    drill-down. Unlike graph_features.py's causal, chronological pass (used
    for leakage-safe model training), this index is built over the *entire*
    dataset -- appropriate for an analyst asking "who is this customer
    connected to, as of now," which is a different question from "what
    would the model have known at transaction time."
    """

    device_to_customers: dict[str, set[str]] = field(default_factory=dict)
    ip_to_customers: dict[str, set[str]] = field(default_factory=dict)
    beneficiary_to_customers: dict[str, set[str]] = field(default_factory=dict)
    customer_has_fraud: dict[str, bool] = field(default_factory=dict)

    def neighbors_of(self, customer_id: str) -> list[tuple[str, list[str]]]:
        """Returns [(neighbor_customer_id, ['device', 'ip', ...]), ...]."""
        shared_via: dict[str, set[str]] = {}
        for mapping, label in (
            (self.device_to_customers, "device"),
            (self.ip_to_customers, "ip"),
            (self.beneficiary_to_customers, "beneficiary"),
        ):
            for group in mapping.values():
                if customer_id in group:
                    for other in group:
                        if other != customer_id:
                            shared_via.setdefault(other, set()).add(label)
        return sorted((cust, sorted(via)) for cust, via in shared_via.items())


@dataclass
class InvestigationStore:
    transactions: pd.DataFrame
    logins: pd.DataFrame
    graph_entities: pd.DataFrame
    model: object
    graph_index: GraphIndex


def _build_graph_index(transactions: pd.DataFrame, graph_entities: pd.DataFrame) -> GraphIndex:
    merged = transactions[["transaction_id", "customer_id", "is_fraud"]].merge(
        graph_entities[["transaction_id", "beneficiary_id", "graph_device_id", "graph_ip_address"]],
        on="transaction_id",
        how="left",
    )
    idx = GraphIndex()
    for row in merged.itertuples(index=False):
        idx.device_to_customers.setdefault(row.graph_device_id, set()).add(row.customer_id)
        idx.ip_to_customers.setdefault(row.graph_ip_address, set()).add(row.customer_id)
        if pd.notna(row.beneficiary_id):
            idx.beneficiary_to_customers.setdefault(row.beneficiary_id, set()).add(row.customer_id)
        if row.is_fraud:
            idx.customer_has_fraud[row.customer_id] = True
    return idx


@lru_cache(maxsize=1)
def get_store() -> InvestigationStore:
    """Load once per process. Raises FileNotFoundError with a clear message
    if `python scripts/run_all.py` hasn't been run yet."""
    import joblib

    for path in (
        config.TRANSACTIONS_WITH_GRAPH_CSV,
        config.LOGIN_EVENTS_CSV,
        config.GRAPH_ENTITIES_CSV,
        config.INVESTIGATION_MODEL_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `python scripts/run_all.py` to generate "
                "Phase 1-3 data and models before using the investigation service."
            )

    transactions = pd.read_csv(config.TRANSACTIONS_WITH_GRAPH_CSV, parse_dates=["timestamp"])
    logins = pd.read_csv(config.LOGIN_EVENTS_CSV, parse_dates=["timestamp"])
    graph_entities = pd.read_csv(config.GRAPH_ENTITIES_CSV)
    model = joblib.load(config.INVESTIGATION_MODEL_PATH)
    graph_index = _build_graph_index(transactions, graph_entities)

    return InvestigationStore(
        transactions=transactions,
        logins=logins,
        graph_entities=graph_entities,
        model=model,
        graph_index=graph_index,
    )


def _row_to_native(row: pd.Series) -> dict:
    """Convert a pandas Series to plain Python types (numpy scalars break
    Pydantic/JSON serialization)."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, pd.Timestamp):
            out[k] = v.round("us").to_pydatetime()
        else:
            out[k] = v
    return out


def get_transaction(transaction_id: str) -> dict:
    store = get_store()
    matches = store.transactions[store.transactions["transaction_id"] == transaction_id]
    if matches.empty:
        raise NotFoundError(f"transaction_id {transaction_id!r} not found")
    return _row_to_native(matches.iloc[0])


def get_auth_history(customer_id: str, before: pd.Timestamp | None = None, limit: int = 20) -> list[dict]:
    store = get_store()
    logins = store.logins[store.logins["customer_id"] == customer_id]
    if before is not None:
        logins = logins[logins["timestamp"] < before]
    logins = logins.sort_values("timestamp", ascending=False).head(limit)
    return [_row_to_native(row) for _, row in logins.iterrows()]


def get_graph_neighbors(customer_id: str) -> list[dict]:
    store = get_store()
    neighbors = store.graph_index.neighbors_of(customer_id)
    return [
        {
            "customer_id": cust,
            "shared_via": via,
            "has_prior_fraud": store.graph_index.customer_has_fraud.get(cust, False),
        }
        for cust, via in neighbors
    ]


def _risk_tier(score: float) -> str:
    if score >= config.RISK_TIER_1_THRESHOLD:
        return "tier_1"
    if score >= config.RISK_TIER_2_THRESHOLD:
        return "tier_2"
    return "tier_3"


def get_risk_score(transaction_id: str, top_n: int = 5) -> dict:
    """Score a transaction and return its top contributing features, using
    XGBoost's native `pred_contribs` (exact per-row SHAP-style
    attribution) -- no extra dependency (e.g. `shap`) required.
    """
    store = get_store()
    row = get_transaction(transaction_id)

    X = pd.DataFrame([row])[config.INVESTIGATION_FEATURE_COLUMNS]
    preprocessor = store.model.named_steps["preprocessor"]
    classifier = store.model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = list(preprocessor.get_feature_names_out())

    proba = float(classifier.predict_proba(X_transformed)[0, 1])

    import xgboost as xgb

    booster = classifier.get_booster()
    dmatrix = xgb.DMatrix(X_transformed, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # last entry is the bias term
    feature_contribs = list(zip(feature_names, contribs[:-1]))

    # Map one-hot-encoded/scaled transformed feature names back to a raw
    # value from the original row where possible, so the explanation is
    # readable (e.g. "merchant_category=wire_transfer", not
    # "categorical__merchant_category_wire_transfer").
    def raw_value_for(transformed_name: str):
        for raw_col in config.INVESTIGATION_FEATURE_COLUMNS:
            if transformed_name.endswith(raw_col) or f"__{raw_col}" in transformed_name:
                return row.get(raw_col)
        return None

    ranked = sorted(feature_contribs, key=lambda fc: abs(fc[1]), reverse=True)[:top_n]
    top_features = [
        {
            "feature": name.split("__", 1)[-1],
            "value": raw_value_for(name),
            "contribution": float(contrib),
        }
        for name, contrib in ranked
    ]

    return {
        "transaction_id": transaction_id,
        "model_version": MODEL_VERSION,
        "risk_score": proba,
        "risk_tier": _risk_tier(proba),
        "top_contributing_features": top_features,
    }


@lru_cache(maxsize=16)
def sample_transactions(n: int = 5, candidate_pool: int = 600) -> list[dict]:
    """Deterministically pick up to `n` deployed transaction_ids with a
    valid risk score, spread across risk tiers where possible -- backs
    `GET /demo/sample-transactions`, so a caller with no prior knowledge of
    this dataset's transaction_ids can immediately try the rest of the API
    (`/transactions/{id}`, `/investigations`, ...) against something real.

    Not the evaluation golden set (see investigation.evaluation.
    select_golden_set for that) -- this just needs *some* valid, varied
    IDs quickly, not a rigorous per-tier sample size. Scans a bounded,
    deterministic candidate pool (every Nth transaction_id in sorted
    order), same shape select_golden_set uses to stay fast without scoring
    the whole dataset. Even so, tier_1 alerts are rare (~1.4% base fraud
    rate, rarer still above the 0.80 threshold -- see POL-AML-001 §2.1), so
    reaching a couple of them can scan hundreds of candidates; cached
    (`lru_cache`) since the answer is deterministic per (n, candidate_pool)
    for the life of the process and this backs a demo endpoint likely hit
    repeatedly -- computing it fresh on every request would be a
    multi-second response for no benefit.
    """
    store = get_store()
    sorted_ids = sorted(store.transactions["transaction_id"].tolist())
    step = max(1, len(sorted_ids) // candidate_pool)
    candidates = sorted_ids[::step][:candidate_pool]

    tier_order = ["tier_1", "tier_2", "tier_3"]
    by_tier: dict[str, list[dict]] = {t: [] for t in tier_order}
    per_tier_cap = -(-n // len(tier_order))  # ceil(n / 3), for tier variety

    for txn_id in candidates:
        risk = get_risk_score(txn_id)
        tier = risk["risk_tier"]
        if len(by_tier[tier]) < per_tier_cap:
            by_tier[tier].append(
                {
                    "transaction_id": risk["transaction_id"],
                    "risk_score": risk["risk_score"],
                    "risk_tier": risk["risk_tier"],
                }
            )
        if sum(len(v) for v in by_tier.values()) >= n:
            break

    samples = [row for tier in tier_order for row in by_tier[tier]]
    return samples[:n]
