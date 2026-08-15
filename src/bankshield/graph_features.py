"""Causal graph-risk feature engineering (Phase 3).

Unlike Phase 1/2's per-customer causal loops, a graph connects customers
*to each other* -- so features here are built with a single chronological
pass over ALL transactions, maintaining incremental graph state (who has
used which device/IP/beneficiary so far, and which customers are known to
have committed fraud so far). For every transaction, in timestamp order:

1. Compute its features from the graph state accumulated so far.
2. *Only afterward* update that state with this transaction's own
   device/IP/beneficiary usage and fraud outcome.

A transaction therefore never sees its own contribution to the graph, or
anything that happens later -- the same causality discipline Phase 1/2
apply to `transaction_velocity_24h` and `cyber_failed_logins_1h`, just
carried out with a global incremental pass instead of a per-customer one
because edges cross customer boundaries.

Features produced (see config.GRAPH_FEATURE_COLUMNS):

- `graph_shared_device_count` -- other customers who used this
  transaction's device before now.
- `graph_shared_ip_count` -- same, for the IP address.
- `graph_beneficiary_connectivity` -- other customers who sent to this
  transaction's beneficiary before now (0 if not a beneficiary-bearing
  category).
- `graph_suspicious_neighbor_count` -- of this customer's known neighbors
  (linked via shared device/IP/beneficiary established before now), how
  many have committed fraud before now.
- `graph_account_network_risk` -- fraud rate among those known neighbors
  (0 if the customer has no neighbors yet).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def add_graph_features(transactions_df: pd.DataFrame, graph_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `transactions_df` with graph_* columns appended."""
    df = transactions_df.merge(
        graph_df[["transaction_id", "beneficiary_id", "graph_device_id", "graph_ip_address"]],
        on="transaction_id",
        how="left",
    )

    order = np.argsort(df["timestamp"].to_numpy(), kind="stable")

    customer_arr = df["customer_id"].to_numpy()
    device_arr = df["graph_device_id"].to_numpy()
    ip_arr = df["graph_ip_address"].to_numpy()
    benef_arr = df["beneficiary_id"].to_numpy()
    fraud_arr = df["is_fraud"].to_numpy()

    n = len(df)
    shared_device_count = np.zeros(n, dtype=int)
    shared_ip_count = np.zeros(n, dtype=int)
    beneficiary_connectivity = np.zeros(n, dtype=int)
    suspicious_neighbor_count = np.zeros(n, dtype=int)
    account_network_risk = np.zeros(n, dtype=float)

    device_users: dict = {}
    ip_users: dict = {}
    beneficiary_senders: dict = {}
    neighbors: dict = defaultdict(set)
    has_fraud: dict = defaultdict(bool)

    for pos in order:
        cust = customer_arr[pos]
        dev = device_arr[pos]
        ip = ip_arr[pos]
        benef = benef_arr[pos]
        has_beneficiary = pd.notna(benef)

        # --- read graph state as it stood strictly before this transaction ---
        # Excluding `cust` itself matters here: a customer reusing their own
        # device/IP/beneficiary must never form a self-loop "neighbor" or
        # inflate their own shared-* counts.
        dev_partners = device_users.get(dev, set()) - {cust}
        shared_device_count[pos] = len(dev_partners)

        ip_partners = ip_users.get(ip, set()) - {cust}
        shared_ip_count[pos] = len(ip_partners)

        benef_partners = (beneficiary_senders.get(benef, set()) - {cust}) if has_beneficiary else set()
        beneficiary_connectivity[pos] = len(benef_partners)

        cust_neighbors = neighbors.get(cust, set())
        if cust_neighbors:
            n_suspicious = sum(1 for nb in cust_neighbors if has_fraud[nb])
            suspicious_neighbor_count[pos] = n_suspicious
            account_network_risk[pos] = n_suspicious / len(cust_neighbors)

        # --- only now fold this transaction's own info into the graph ---
        for partner in dev_partners:
            neighbors[cust].add(partner)
            neighbors[partner].add(cust)
        device_users.setdefault(dev, set()).add(cust)

        for partner in ip_partners:
            neighbors[cust].add(partner)
            neighbors[partner].add(cust)
        ip_users.setdefault(ip, set()).add(cust)

        if has_beneficiary:
            for partner in benef_partners:
                neighbors[cust].add(partner)
                neighbors[partner].add(cust)
            beneficiary_senders.setdefault(benef, set()).add(cust)

        if fraud_arr[pos]:
            has_fraud[cust] = True

    df["graph_shared_device_count"] = shared_device_count
    df["graph_shared_ip_count"] = shared_ip_count
    df["graph_beneficiary_connectivity"] = beneficiary_connectivity
    df["graph_suspicious_neighbor_count"] = suspicious_neighbor_count
    df["graph_account_network_risk"] = account_network_risk

    return df.drop(columns=["beneficiary_id", "graph_device_id", "graph_ip_address"])
