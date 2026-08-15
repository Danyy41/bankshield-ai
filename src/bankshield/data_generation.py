"""Synthetic banking transaction data generator.

Design principles
------------------
1. **Causal, two-stage generation.** For every transaction we first realize
   raw behavioural features (device used, country, hour, amount, ...) from
   population-level base rates that do *not* depend on the eventual fraud
   label. Only afterwards do we compute a fraud probability from those
   realized features and sample the label. This mirrors how fraud actually
   happens (the label is a consequence of behaviour, not the cause of it)
   and avoids accidentally baking the target into the features.
2. **Everything a feature depends on is only ever computed from that
   customer's own past.** Device familiarity, beneficiary familiarity,
   rolling transaction velocity, and the customer's historical average
   amount are all computed by walking each customer's transactions in
   timestamp order and only looking backwards. This is what "no obvious
   leakage" means at the data-generation stage, before we even get to the
   train/test split.
3. **Fraud is rare and probabilistic, not a deterministic rule.** The fraud
   probability is a logistic combination of risk signals plus Gaussian
   noise, so fraudulent and legitimate transactions have overlapping
   feature distributions -- fraud is statistically distinguishable, not
   trivially separable.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from faker import Faker

from bankshield import config


def _calibrate_intercept(base_logit: np.ndarray, target_rate: float) -> float:
    """Binary-search the logistic intercept that yields `target_rate` on average."""

    def mean_rate(b0: float) -> float:
        return _sigmoid(base_logit + b0).mean()

    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_rate(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _split_transaction_counts(n_customers: int, n_transactions: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each customer a transaction count (>=1) summing exactly to n_transactions."""
    mean_per_customer = n_transactions / n_customers
    counts = rng.poisson(lam=max(mean_per_customer - 1, 0.1), size=n_customers) + 1
    diff = n_transactions - counts.sum()
    # Nudge random customers up/down until the total matches exactly.
    idx_cycle = rng.permutation(n_customers)
    i = 0
    while diff != 0:
        j = idx_cycle[i % n_customers]
        if diff > 0:
            counts[j] += 1
            diff -= 1
        elif counts[j] > 1:
            counts[j] -= 1
            diff += 1
        i += 1
    return counts


CATEGORY_AMOUNT_PARAMS = {
    # category -> (lognormal mean of log(amount), sigma)
    "grocery": (3.4, 0.5),
    "restaurant": (3.0, 0.6),
    "fuel": (3.5, 0.4),
    "utilities": (4.0, 0.4),
    "online_retail": (3.8, 0.8),
    "electronics": (5.3, 0.9),
    "travel": (5.8, 0.8),
    "healthcare": (4.5, 0.7),
    "entertainment": (3.6, 0.6),
    "cash_withdrawal": (4.6, 0.7),
    "wire_transfer": (5.9, 1.1),
    "gambling": (4.4, 1.0),
    "crypto_exchange": (5.6, 1.2),
}

CATEGORY_WEIGHTS = {
    "grocery": 0.20,
    "restaurant": 0.16,
    "fuel": 0.10,
    "utilities": 0.08,
    "online_retail": 0.14,
    "electronics": 0.06,
    "travel": 0.04,
    "healthcare": 0.05,
    "entertainment": 0.06,
    "cash_withdrawal": 0.05,
    "wire_transfer": 0.03,
    "gambling": 0.02,
    "crypto_exchange": 0.01,
}

# Hourly transaction weights (index = hour 0-23): fewer transactions overnight.
HOUR_WEIGHTS = np.array(
    [
        0.010, 0.007, 0.005, 0.004, 0.004, 0.006,  # 00-05
        0.015, 0.030, 0.050, 0.060, 0.060, 0.065,  # 06-11
        0.070, 0.065, 0.060, 0.058, 0.058, 0.062,  # 12-17
        0.068, 0.060, 0.045, 0.030, 0.020, 0.015,  # 18-23
    ]
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


def generate_transactions(
    n_transactions: int = config.N_TRANSACTIONS,
    n_customers: int = config.N_CUSTOMERS,
    target_fraud_rate: float = config.TARGET_FRAUD_RATE,
    simulation_days: int = config.SIMULATION_DAYS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic banking transactions dataset.

    Returns a DataFrame sorted by timestamp with the columns documented in
    the Phase 1 README.
    """
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    sim_start = pd.Timestamp("2026-01-01")
    sim_end = sim_start + pd.Timedelta(days=simulation_days)

    categories = list(CATEGORY_WEIGHTS)
    category_p = np.array([CATEGORY_WEIGHTS[c] for c in categories])

    # ---- customer profiles ------------------------------------------------
    customer_ids = [f"CUST{idx:06d}" for idx in range(n_customers)]
    home_countries = rng.choice(config.HOME_COUNTRIES, size=n_customers)
    # Most accounts pre-date the simulation window by a while; a tail of
    # accounts is brand new (opened right around simulation start).
    account_age_at_start_days = rng.exponential(scale=750, size=n_customers).clip(0, 3650)
    account_created_at = sim_start - pd.to_timedelta(account_age_at_start_days, unit="D")
    # Per-customer spending-level multiplier (some customers just spend more).
    spend_multiplier = rng.lognormal(mean=0.0, sigma=0.35, size=n_customers)
    # Each customer starts the simulation with exactly one known device
    # (their existing phone/laptop) so day-one transactions aren't all
    # flagged as a new device.
    primary_device = [f"dev_{uuid.uuid4().hex[:10]}" for _ in range(n_customers)]

    txn_counts = _split_transaction_counts(n_customers, n_transactions, rng)

    rows = []
    for c_idx in range(n_customers):
        n = int(txn_counts[c_idx])
        cust_id = customer_ids[c_idx]
        home_country = home_countries[c_idx]
        created_at = account_created_at[c_idx]
        mult = spend_multiplier[c_idx]

        # Random timestamps within the simulation window, sorted so we can
        # walk this customer's history causally.
        offsets_seconds = rng.uniform(0, simulation_days * 86400, size=n)
        days = np.floor(offsets_seconds / 86400).astype(int)
        hours = rng.choice(24, size=n, p=HOUR_WEIGHTS)
        minutes = rng.integers(0, 60, size=n)
        seconds = rng.integers(0, 60, size=n)
        timestamps = sim_start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(
            hours, unit="h"
        ) + pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(seconds, unit="s")
        order = np.argsort(timestamps.values)
        timestamps = timestamps[order]

        categories_c = rng.choice(categories, size=n, p=category_p)

        known_devices = {primary_device[c_idx]}
        known_beneficiaries: set[str] = set()
        prior_amount_sum, prior_amount_count = 0.0, 0
        recent_ts: list[pd.Timestamp] = []

        for i in range(n):
            ts = timestamps[i]
            category = categories_c[i]

            # --- amount ---
            log_mu, log_sigma = CATEGORY_AMOUNT_PARAMS[category]
            amount = float(rng.lognormal(mean=log_mu, sigma=log_sigma) * mult)
            amount = round(min(amount, 25_000.0), 2)

            # --- device: baseline 3.5% chance of a never-seen-before device ---
            if known_devices and rng.random() > 0.035:
                device_id = rng.choice(list(known_devices))
                new_device = False
            else:
                device_id = f"dev_{uuid.uuid4().hex[:10]}"
                new_device = device_id not in known_devices
            known_devices.add(device_id)

            # --- country: baseline 5% chance of a foreign transaction ---
            if rng.random() < 0.05:
                foreign_choices = [c for c in config.FOREIGN_COUNTRIES if c != home_country]
                txn_country = rng.choice(foreign_choices)
            else:
                txn_country = home_country
            country_mismatch = txn_country != home_country

            # --- beneficiary: only meaningful for transfer-like categories ---
            if category in ("wire_transfer", "cash_withdrawal"):
                if known_beneficiaries and rng.random() > 0.25:
                    beneficiary = rng.choice(list(known_beneficiaries))
                    new_beneficiary = False
                else:
                    beneficiary = f"benef_{uuid.uuid4().hex[:8]}"
                    new_beneficiary = True
                known_beneficiaries.add(beneficiary)
            else:
                new_beneficiary = False

            # --- rolling 24h transaction velocity (causal, this customer only) ---
            recent_ts.append(ts)
            cutoff = ts - pd.Timedelta(hours=24)
            recent_ts = [t for t in recent_ts if t >= cutoff]
            velocity_24h = len(recent_ts)

            # --- amount vs this customer's own prior history ---
            prior_avg = (prior_amount_sum / prior_amount_count) if prior_amount_count > 0 else amount
            amount_to_avg_ratio = amount / prior_avg if prior_avg > 0 else 1.0
            prior_amount_sum += amount
            prior_amount_count += 1

            account_age_days = max((ts - created_at).days, 0)
            hour_of_day = int(ts.hour)
            day_of_week = int(ts.dayofweek)
            is_night = hour_of_day < 6

            rows.append(
                {
                    "transaction_id": uuid.uuid4().hex,
                    "customer_id": cust_id,
                    "timestamp": ts,
                    "amount": amount,
                    "merchant_category": category,
                    "home_country": home_country,
                    "country": txn_country,
                    "device_id": device_id,
                    "ip_address": faker.ipv4_public(),
                    "account_age_days": account_age_days,
                    "transaction_velocity_24h": velocity_24h,
                    "new_device": new_device,
                    "new_beneficiary": new_beneficiary,
                    "country_mismatch": country_mismatch,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "is_night": is_night,
                    "amount_to_avg_ratio": round(amount_to_avg_ratio, 3),
                }
            )

    df = pd.DataFrame(rows)

    # ---- fraud label: logistic combination of risk signals + noise --------
    high_risk_category = df["merchant_category"].isin(config.HIGH_RISK_CATEGORIES).to_numpy()
    new_account = (df["account_age_days"] < 14).to_numpy()
    high_velocity = (df["transaction_velocity_24h"] >= 4).to_numpy()
    amount_anomaly = np.log1p(df["amount_to_avg_ratio"].to_numpy())

    noise = rng.normal(loc=0.0, scale=1.4, size=len(df))

    new_device_arr = df["new_device"].to_numpy()
    country_mismatch_arr = df["country_mismatch"].to_numpy()
    is_night_arr = df["is_night"].to_numpy()

    # A handful of genuine (non-additive) interaction effects: certain risk
    # factors are disproportionately more dangerous *in combination* than
    # the sum of their individual effects would suggest (e.g. a stolen-card
    # tester using an unfamiliar device at 3am, or a cross-border transfer
    # into a high-risk category). Linear models can only approximate this
    # with the individual terms; tree-based models can learn it directly.
    stolen_card_pattern = new_device_arr & is_night_arr
    laundering_pattern = country_mismatch_arr & high_risk_category

    base_logit = (
        2.2 * new_device_arr
        + 1.0 * df["new_beneficiary"].to_numpy()
        + 1.8 * country_mismatch_arr
        + 1.1 * is_night_arr
        + 1.3 * high_risk_category
        + 0.9 * amount_anomaly
        + 0.8 * high_velocity
        + 1.5 * new_account
        + 1.6 * stolen_card_pattern
        + 1.4 * laundering_pattern
        + noise
    )
    intercept = _calibrate_intercept(base_logit, target_fraud_rate)
    fraud_prob = _sigmoid(base_logit + intercept)
    df["is_fraud"] = rng.binomial(1, fraud_prob).astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)

    ordered_cols = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "merchant_category",
        "home_country",
        "country",
        "country_mismatch",
        "device_id",
        "new_device",
        "ip_address",
        "account_age_days",
        "transaction_velocity_24h",
        "new_beneficiary",
        "hour_of_day",
        "day_of_week",
        "is_night",
        "amount_to_avg_ratio",
        "is_fraud",
    ]
    return df[ordered_cols]


if __name__ == "__main__":
    data = generate_transactions()
    print(data.shape)
    print(data["is_fraud"].mean())
