"""Synthetic graph-relationship generator (Phase 3).

Reads Phase 1's `transactions.csv` (with its already-fixed `is_fraud`
labels) and never modifies it -- exactly the pattern Phase 2 used for
login events. Two things happen here:

1. **Beneficiary identity reconstruction.** Phase 1 only persisted a
   `new_beneficiary` boolean, not an actual beneficiary ID. This module
   reconstructs a beneficiary identity space that is *consistent* with
   that flag (a customer's first "new beneficiary" transaction gets a
   fresh ID, later non-new ones reuse one of their own prior IDs) so
   graph features have something real to connect through. These IDs are
   new synthetic labels generated here, not literally recovered from
   Phase 1's internal generation state.
2. **Mule-ring injection.** Independently-randomized device IDs, IPs, and
   (reconstructed) beneficiary IDs essentially never collide by chance,
   so realistic shared-infrastructure structure has to be deliberately
   simulated: a handful of rings, each a small group of customers who
   share a device/IP/beneficiary pool. Membership is *skewed toward*
   customers who already have a fraud transaction (not guaranteed --
   rings pull in a few incidental/innocent members too), and only a
   fraction of each member's transactions (biased toward their
   fraudulent ones) actually get the shared identifiers -- so ring
   membership correlates with fraud without being a deterministic tell.

Everything here is layered onto NEW columns (`graph_device_id`,
`graph_ip_address`, `beneficiary_id`) that default to a transaction's own
real device/IP -- the original `device_id`/`ip_address`/`new_device`
columns Phase 1/2 rely on are untouched.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from faker import Faker

from bankshield import config

BENEFICIARY_CATEGORIES = ("wire_transfer", "cash_withdrawal")


def _reconstruct_beneficiary_ids(transactions_df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    beneficiary_id = pd.Series([None] * len(transactions_df), index=transactions_df.index, dtype=object)

    eligible_mask = transactions_df["merchant_category"].isin(BENEFICIARY_CATEGORIES)
    eligible = transactions_df.loc[eligible_mask].sort_values("timestamp")

    for _customer_id, group in eligible.groupby("customer_id"):
        known: list[str] = []
        for idx, is_new in zip(group.index, group["new_beneficiary"]):
            if is_new or not known:
                new_id = f"benef_{uuid.uuid4().hex[:10]}"
                known.append(new_id)
                beneficiary_id.at[idx] = new_id
            else:
                beneficiary_id.at[idx] = rng.choice(known)

    return beneficiary_id


def _inject_mule_rings(
    transactions_df: pd.DataFrame,
    beneficiary_id: pd.Series,
    graph_device_id: pd.Series,
    graph_ip_address: pd.Series,
    rng: np.random.Generator,
    faker: Faker,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    fraud_customers = transactions_df.loc[transactions_df["is_fraud"] == 1, "customer_id"].unique()
    all_customers = transactions_df["customer_id"].unique()

    ring_id = pd.Series([None] * len(transactions_df), index=transactions_df.index, dtype=object)
    is_ring_member = pd.Series(False, index=transactions_df.index)

    for ring_num in range(config.N_RINGS):
        size = int(rng.integers(config.RING_SIZE_MIN, config.RING_SIZE_MAX + 1))
        members = set()
        for _ in range(size):
            if len(fraud_customers) > 0 and rng.random() < config.RING_FRAUD_MEMBER_BIAS:
                members.add(rng.choice(fraud_customers))
            else:
                members.add(rng.choice(all_customers))

        shared_devices = [f"dev_ring_{uuid.uuid4().hex[:8]}" for _ in range(int(rng.integers(2, 4)))]
        shared_ips = [faker.ipv4_public() for _ in range(int(rng.integers(2, 4)))]
        shared_beneficiaries = [f"benef_ring_{uuid.uuid4().hex[:8]}" for _ in range(int(rng.integers(1, 3)))]
        ring_name = f"ring_{ring_num:03d}"

        # Iterate in a fixed order, not the set's native order: Python
        # randomizes str hash seeds per-process, so iterating `members`
        # directly would consume `rng` in a different sequence on every
        # run and silently break reproducibility across process runs.
        for customer_id in sorted(members):
            cust_idx = transactions_df.index[transactions_df["customer_id"] == customer_id]
            is_ring_member.loc[cust_idx] = True
            ring_id.loc[cust_idx] = ring_name

            for idx in cust_idx:
                override_prob = config.RING_TRANSACTION_OVERRIDE_RATE * (
                    2.0 if transactions_df.at[idx, "is_fraud"] else 1.0
                )
                if rng.random() < min(override_prob, 0.9):
                    graph_device_id.at[idx] = rng.choice(shared_devices)
                    graph_ip_address.at[idx] = rng.choice(shared_ips)

                if beneficiary_id.at[idx] is not None and rng.random() < config.RING_BENEFICIARY_OVERRIDE_RATE:
                    beneficiary_id.at[idx] = rng.choice(shared_beneficiaries)

    return beneficiary_id, graph_device_id, graph_ip_address, ring_id, is_ring_member


def generate_graph_relationships(
    transactions_df: pd.DataFrame,
    seed: int = config.GRAPH_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate the Phase 3 graph-entity layer for `transactions_df`.

    Returns a DataFrame indexed like `transactions_df` with columns:
    transaction_id, customer_id, beneficiary_id, graph_device_id,
    graph_ip_address, ring_id, is_ring_member (the last two are
    generation-time diagnostics, never used as model features).
    """
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    beneficiary_id = _reconstruct_beneficiary_ids(transactions_df, rng)
    graph_device_id = transactions_df["device_id"].copy()
    graph_ip_address = transactions_df["ip_address"].copy()

    beneficiary_id, graph_device_id, graph_ip_address, ring_id, is_ring_member = _inject_mule_rings(
        transactions_df, beneficiary_id, graph_device_id, graph_ip_address, rng, faker
    )

    return pd.DataFrame(
        {
            "transaction_id": transactions_df["transaction_id"],
            "customer_id": transactions_df["customer_id"],
            "beneficiary_id": beneficiary_id,
            "graph_device_id": graph_device_id,
            "graph_ip_address": graph_ip_address,
            "ring_id": ring_id,
            "is_ring_member": is_ring_member,
        }
    )
