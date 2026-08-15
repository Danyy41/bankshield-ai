# POL-CASE-004 — Case Management, Escalation & Human-Approval Policy

**Document type:** Operations / governance policy
**Version:** 1.5
**Effective date:** 2026-02-01
**Owner:** BankShield Financial Crime Operations
**Applies to:** All case creation, whether initiated by a human analyst or
an AI-assisted investigation tool

## §1. Purpose

This policy defines what constitutes a "case," how cases are created and
escalated, and — critically for AI-assisted investigation — which actions
are consequential enough to require explicit human approval before they
take effect.

## §2. Case lifecycle

1. **Draft** — a proposed case with supporting evidence, created by either
   a human analyst or an AI investigation assistant. A draft has no
   operational effect: no account restriction, no customer contact, no
   regulatory filing.
2. **Pending approval** — a draft that has been submitted for human review.
3. **Approved** — a human reviewer with case-approval authority has
   confirmed the evidence and disposition. Only an approved case may trigger
   downstream actions (SAR referral per POL-AML-001 §5, account restriction,
   customer notification per POL-ATO-003 §4).
4. **Rejected** — a human reviewer has determined the draft does not meet
   the evidentiary bar. The rejection reason must be recorded.
5. **Closed** — the case has been fully dispositioned and downstream actions
   (if any) completed.

## §3. Consequential actions require human approval

The following are always **consequential actions** and must never be
executed automatically by an AI system, regardless of how confident the
model's output is:

1. Creating a case that names a specific customer as a suspect of fraud or
   money laundering.
2. Any account restriction, hold, or session termination.
3. Any regulatory filing (SAR or equivalent).
4. Any customer-facing communication asserting a fraud/AML determination.

An AI investigation assistant may **propose** any of the above — producing a
structured draft with cited evidence — but the system must present that
proposal to a human reviewer and receive explicit, recorded approval before
the action is carried out. A tool call that would create a case is only
ever a request for the case to be created; execution is gated on approval.

## §4. Approval record requirements

Every approval decision must record: the reviewer's identity, the timestamp,
the evidence considered (by reference to the case's cited evidence), and —
for a rejection — the specific reason. This record is part of the audit
trail for both fraud operations and regulatory examination.

## §5. Escalation SLAs

- Tier 1 alerts (POL-AML-001 §2.1): case must reach "pending approval" within
  4 business hours of alert generation.
- Tier 2 alerts: within 1 business day.
- An AI-assisted investigation that cannot reach a confident disposition
  (e.g., contradictory evidence, missing data) must be escalated to a human
  analyst rather than defaulting to either "confirmed fraud" or "false
  positive."

## §6. Analyst override

A human reviewer may reject or modify any AI-generated recommendation. The
AI system's role is to accelerate evidence-gathering and produce a
well-cited draft, not to make the final determination.
