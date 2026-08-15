"""Synthetic login/session event generator (Phase 2).

Reads Phase 1's already-generated `transactions.csv` (with its fixed
`is_fraud` labels) and never modifies it -- login events are generated
*from* the transaction history, not the other way around, so Phase 1's
data, model, and metrics are completely unaffected by anything here.

Design
------
Every transaction gets a "pre-auth" login session shortly before it
(logging in is a precondition of transacting). What that session looks
like depends on the transaction:

- **Fraudulent transactions** are preceded by an account-takeover (ATO)
  login pattern most of the time (`ATO_INJECTION_RATE_FRAUD`): a burst of
  failed attempts from unfamiliar devices/countries, ending in one
  success from a new device, often from an unusual country.
- **Legitimate transactions** almost always get a single ordinary login
  from a familiar device/country, but a small fraction
  (`ATO_FALSE_ALARM_RATE`) still show a benign ATO-like burst -- someone
  really did mistype their password a few times, or log in from a new
  phone, without anything fraudulent happening. This keeps cyber
  telemetry a *statistical* signal, not a deterministic tell.

On top of that, each customer gets a handful of standalone "routine"
logins (checking a balance, etc.) unrelated to any transaction.

This mirrors how account takeover actually works causally (compromise
the login, then transact) rather than assigning login patterns to
fraud after the fact, so the resulting correlation is genuine signal
for a model to find, not leakage.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from faker import Faker

from bankshield import config


def _sample_session_duration(rng: np.random.Generator) -> float:
    return float(np.clip(rng.lognormal(mean=np.log(180), sigma=0.9), 10, 3600))


def generate_login_events(
    transactions_df: pd.DataFrame,
    seed: int = config.AUTH_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate login/session events for the customers in `transactions_df`."""
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    global_start = transactions_df["timestamp"].min()
    global_end = transactions_df["timestamp"].max()
    global_span_seconds = (global_end - global_start).total_seconds()

    home_country_by_customer = transactions_df.groupby("customer_id")["home_country"].first()

    rows: list[dict] = []

    for customer_id, home_country in home_country_by_customer.items():
        cust_txns = transactions_df.loc[
            transactions_df["customer_id"] == customer_id, ["timestamp", "is_fraud"]
        ].sort_values("timestamp")

        # --- anchors: one per transaction (pre-auth login) + routine logins ---
        anchors = []  # list of (timestamp, kind, is_fraud)
        for ts, is_fraud in cust_txns.itertuples(index=False):
            lead_seconds = rng.uniform(30, 1200)
            anchors.append((ts - pd.Timedelta(seconds=lead_seconds), "transaction", bool(is_fraud)))

        n_routine = rng.poisson(config.ROUTINE_LOGINS_PER_CUSTOMER_MEAN)
        for _ in range(n_routine):
            offset_seconds = rng.uniform(0, global_span_seconds)
            anchors.append((global_start + pd.Timedelta(seconds=offset_seconds), "routine", False))

        anchors.sort(key=lambda a: a[0])

        # Each customer starts with one familiar login device (their phone/laptop).
        known_devices = {f"dev_{uuid.uuid4().hex[:10]}"}

        for anchor_ts, kind, is_fraud in anchors:
            if kind == "transaction" and is_fraud:
                is_ato = rng.random() < config.ATO_INJECTION_RATE_FRAUD
            else:
                is_ato = rng.random() < config.ATO_FALSE_ALARM_RATE

            session_id = uuid.uuid4().hex

            if is_ato:
                n_failed = int(rng.integers(2, 7))
                t = anchor_ts - pd.Timedelta(seconds=rng.uniform(60, 600))
                for _ in range(n_failed):
                    attacker_device = f"dev_{uuid.uuid4().hex[:10]}"
                    attacker_country = rng.choice(
                        [c for c in config.FOREIGN_COUNTRIES if c != home_country]
                    )
                    rows.append(
                        {
                            "login_id": uuid.uuid4().hex,
                            "session_id": session_id,
                            "customer_id": customer_id,
                            "timestamp": t,
                            "device_id": attacker_device,
                            "ip_address": faker.ipv4_public(),
                            "country": attacker_country,
                            "login_success": False,
                            "new_device": attacker_device not in known_devices,
                            "new_location": attacker_country != home_country,
                            "session_duration_seconds": 0.0,
                            "is_ato_pattern": True,
                        }
                    )
                    t = t + pd.Timedelta(seconds=rng.uniform(5, 90))

                # Final successful login: usually a new device, often a new location.
                success_device = (
                    f"dev_{uuid.uuid4().hex[:10]}" if rng.random() < 0.90 else rng.choice(list(known_devices))
                )
                success_country = (
                    rng.choice([c for c in config.FOREIGN_COUNTRIES if c != home_country])
                    if rng.random() < 0.80
                    else home_country
                )
                rows.append(
                    {
                        "login_id": uuid.uuid4().hex,
                        "session_id": session_id,
                        "customer_id": customer_id,
                        "timestamp": anchor_ts,
                        "device_id": success_device,
                        "ip_address": faker.ipv4_public(),
                        "country": success_country,
                        "login_success": True,
                        "new_device": success_device not in known_devices,
                        "new_location": success_country != home_country,
                        "session_duration_seconds": _sample_session_duration(rng),
                        "is_ato_pattern": True,
                    }
                )
                known_devices.add(success_device)

            else:
                t = anchor_ts
                # Occasionally a single mistyped-password retry before success.
                if rng.random() < 0.05:
                    t = t - pd.Timedelta(seconds=rng.uniform(5, 30))
                    same_device = rng.choice(list(known_devices))
                    rows.append(
                        {
                            "login_id": uuid.uuid4().hex,
                            "session_id": session_id,
                            "customer_id": customer_id,
                            "timestamp": t,
                            "device_id": same_device,
                            "ip_address": faker.ipv4_public(),
                            "country": home_country,
                            "login_success": False,
                            "new_device": False,
                            "new_location": False,
                            "session_duration_seconds": 0.0,
                            "is_ato_pattern": False,
                        }
                    )

                device = (
                    f"dev_{uuid.uuid4().hex[:10]}" if rng.random() < 0.03 else rng.choice(list(known_devices))
                )
                country = (
                    rng.choice([c for c in config.FOREIGN_COUNTRIES if c != home_country])
                    if rng.random() < 0.05
                    else home_country
                )
                rows.append(
                    {
                        "login_id": uuid.uuid4().hex,
                        "session_id": session_id,
                        "customer_id": customer_id,
                        "timestamp": anchor_ts,
                        "device_id": device,
                        "ip_address": faker.ipv4_public(),
                        "country": country,
                        "login_success": True,
                        "new_device": device not in known_devices,
                        "new_location": country != home_country,
                        "session_duration_seconds": _sample_session_duration(rng),
                        "is_ato_pattern": False,
                    }
                )
                known_devices.add(device)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    ordered_cols = [
        "login_id",
        "session_id",
        "customer_id",
        "timestamp",
        "device_id",
        "ip_address",
        "country",
        "login_success",
        "new_device",
        "new_location",
        "session_duration_seconds",
        "is_ato_pattern",
    ]
    return df[ordered_cols]
