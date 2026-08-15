# POL-KYC-005 — Customer Due Diligence & High-Risk Category Policy

**Document type:** KYC / due-diligence policy
**Version:** 2.0
**Effective date:** 2025-08-15
**Owner:** BankShield Compliance
**Applies to:** Onboarding and ongoing due diligence for all deposit account
customers

## §1. Purpose

Defines the customer- and merchant-category risk factors that feed into
BankShield's overall risk posture for an account, independent of any single
transaction's fraud score.

## §2. High-risk merchant categories

The following categories are designated high-risk for money-laundering and
fraud purposes, and any transaction in these categories receives additional
scrutiny per POL-AML-001 §3:

- `gambling`
- `crypto_exchange`
- `wire_transfer`
- `cash_withdrawal`

## §3. Account-age risk factor

New accounts (`account_age_days` below 30) transacting in a high-risk
category, especially combined with a new device or new beneficiary, warrant
closer review than an equivalent transaction on a long-established account.
Account age alone is never sufficient grounds for escalation — it is a
contextual multiplier on other signals.

## §4. Cross-border activity

A transaction where `country` differs from the customer's `home_country`
(`country_mismatch = true`) is not inherently suspicious — many customers
travel or transact internationally as a matter of course. It becomes a
meaningful risk factor only in combination with other signals per
POL-FRAUD-002 §3.1 (e.g., cross-border plus new beneficiary, or cross-border
plus high-risk category).

## §5. Ongoing monitoring obligation

Every account, regardless of prior clean history, remains subject to
ongoing transaction monitoring per POL-AML-001. A clean prior history is
relevant context for an investigation but does not exempt a transaction from
the standard alert-tier process.
