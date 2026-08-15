# POL-GRAPH-006 — Mule Network & Money-Laundering Ring Typology Reference

**Document type:** AML / financial-crime intelligence reference
**Version:** 1.3
**Effective date:** 2025-12-01
**Owner:** BankShield Financial Crime Compliance
**Applies to:** Graph/network signals produced by the Phase 3 graph
intelligence pipeline (`graph_*` features)

## §1. Purpose

This reference describes how BankShield's graph-relationship features map to
known money-mule and layering typologies, so analysts and AI-assisted
investigations interpret network signals consistently with regulatory
expectations.

## §2. Graph feature reference

- `graph_shared_device_count` — number of other customers who used this
  transaction's device before now.
- `graph_shared_ip_count` — same, for IP address.
- `graph_beneficiary_connectivity` — number of other customers who have sent
  funds to this transaction's beneficiary before now.
- `graph_suspicious_neighbor_count` — of this customer's known network
  neighbors (linked via shared device/IP/beneficiary), how many have a prior
  confirmed-fraud transaction.
- `graph_account_network_risk` — fraud rate among those known neighbors, in
  [0, 1].

## §3. Mule-ring typology

A "mule ring" is a set of accounts, often onboarded independently but
controlled or coordinated by the same actor(s), used to receive, layer, and
move fraudulent or laundered funds. BankShield's historical mule rings share
these characteristics:

1. **Size**: typically 3–7 linked customer accounts.
2. **Shared infrastructure**: a small pool of devices, IP addresses, and/or
   beneficiary accounts reused across multiple ring members' transactions.
3. **Elevated fraud rate**: ring members show a fraud rate roughly 8–10x the
   general population baseline, though most individual transactions by ring
   members are *not* fraudulent — the ring's structure, not every
   transaction, is the tell.
4. **Partial reuse**: shared infrastructure is typically reused for a
   fraction of a member's transactions, not all of them — a ring member also
   has ordinary, unrelated activity.

## §4. Interpreting network signals for an individual alert

A nonzero `graph_shared_device_count` or `graph_shared_ip_count` alone is
**weak evidence** — device/IP sharing can be innocuous (a shared household
device, a public Wi-Fi network, or coincidence). It becomes meaningful
network evidence when combined with:

- `graph_suspicious_neighbor_count > 0` (a linked account has confirmed
  prior fraud), **and/or**
- `graph_account_network_risk` materially above the population baseline
  fraud rate (~1.3–1.5% in BankShield's transaction population).

When both conditions hold, treat the transaction as touching a probable
mule-ring structure and escalate per POL-AML-001 §4.

## §5. Beneficiary-connectivity specifically

`graph_beneficiary_connectivity > 0` — multiple customers sending funds to
the same beneficiary — is a classic layering/mule indicator, especially in
the `wire_transfer` category. Combined with `new_beneficiary = true` on the
current transaction, this pattern should be cited explicitly in any case
file recommending SAR referral.

## §6. Evidentiary caution

Network signals describe *structure*, not intent. A shared device or IP does
not by itself establish that a customer is a knowing participant in a mule
ring — case files must distinguish "this transaction touches network
structure consistent with known mule-ring typology" from "this customer is a
money mule," and leave the latter determination to human reviewers per
POL-CASE-004 §6.
