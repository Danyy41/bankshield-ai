# POL-ATO-003 — Account Takeover Detection & Response Procedure

**Document type:** Cybersecurity / fraud operations policy
**Version:** 2.1
**Effective date:** 2025-09-01
**Owner:** BankShield Cyber-Fraud Fusion Team
**Applies to:** Login/session telemetry and any transaction preceded by
authentication activity within the lookback window defined in §2

## §1. Purpose

Account takeover (ATO) is when a fraudster gains access to a legitimate
customer's account credentials or session and transacts as that customer.
This procedure defines the authentication-signal shape BankShield treats as
indicative of ATO and how it should be combined with transaction risk.

## §2. The ATO signal shape

BankShield's cyber-telemetry features summarize each customer's login
history in the 24 hours before a transaction:

- `cyber_failed_logins_1h` — failed attempts in the hour before the
  transaction.
- `cyber_login_count_24h` — total login attempts in the trailing 24h.
- `cyber_minutes_since_last_login` — recency of the most recent login.
- `cyber_new_device_recent` / `cyber_unusual_country_recent` — whether the
  most recent login was from an unfamiliar device or location.
- `cyber_recent_suspicious_auth` — composite flag: several recent failed
  attempts *and* a suspicious final success (new device or new location).

### §2.1 The canonical ATO pattern

The pattern most strongly associated with confirmed ATO at BankShield is:
**a burst of 2 or more failed login attempts within the hour before the
transaction, followed by a successful login from a new device or an
unfamiliar country.** This is exactly what `cyber_recent_suspicious_auth`
captures. Analysts should treat `cyber_recent_suspicious_auth = true` as the
single highest-priority cyber signal in an investigation — in BankShield's
historical data this feature has the strongest correlation with confirmed
fraud of any cyber feature.

### §2.2 Signals that are weaker in isolation

A single new-device login (`cyber_new_device_recent = true`) without recent
failed attempts is common and often benign (a customer got a new phone).
Do not treat a lone new-device flag as sufficient grounds for escalation —
require it in combination with either failed-attempt history or a
transaction-side risk signal (high-risk category, cross-border, velocity
spike) per POL-FRAUD-002 §3.1.

## §3. Response actions by severity

| Cyber signal | Transaction risk | Required action |
|---|---|---|
| `cyber_recent_suspicious_auth = true` | Any | Tier 1 handling regardless of the model's standalone score; treat as probable ATO |
| New device only, no failed attempts | High-risk category or cross-border | Tier 2 handling; verify device history before disposition |
| No suspicious cyber signal | Any | Handle at the tier indicated by the transaction risk score alone |

## §4. Customer notification and account action

If ATO is confirmed, the account's active sessions must be terminated and
the customer notified through an out-of-band channel (not the potentially
compromised device/session). **Session termination and any account
restriction are consequential actions and require human approval** per
POL-CASE-004 §3 before execution — an AI-assisted investigation may
recommend this action but must not execute it.

## §5. Evidentiary standard for ATO findings

A finding that a transaction is likely ATO-driven must cite the specific
cyber_* feature values that support it (not just the composite flag) and
the transaction-side facts (device ID, IP, country) so a human reviewer can
independently verify the claim against the raw authentication log via
`get_auth_history`.
