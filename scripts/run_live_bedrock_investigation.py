"""Run ONE real fraud investigation on Amazon Bedrock and, if it passes
every check, save a sanitized copy as the README's live example.

THIS MAKES BILLABLE BEDROCK CALLS. It refuses to run without
--confirm-billable. One investigation is 3-8 model turns; with Claude
Opus 5.5 that is typically ~$0.10-0.40 and at most ~$2 (8 turns at the
8,192-token output cap).

What it does:
1. runs the free preflight (scripts/bedrock_preflight.py) and stops if blocked
2. picks a Tier 1 alert (or --transaction-id) and runs the unchanged agent
   loop -- tools through the guardrails, citations filtered to what was
   retrieved -- against BedrockClaudeClient
3. verifies: real model, tools used without errors, at least one verified
   citation and no unverified ones, and any proposed case is still only a
   pending approval (no case created)
4. then has a separate reviewer approve the proposal, showing the case is
   created only by that human decision
5. writes docs/examples/live_bedrock_investigation.{json,md} with AWS
   identifiers (ARNs, account IDs, access keys, request IDs) redacted

All state goes to a private in-memory database -- nothing touches the
configured BankShield database.

Usage:
    python scripts/run_live_bedrock_investigation.py --confirm-billable [--transaction-id T]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["BANKSHIELD_DATABASE_URL"] = "sqlite://"

from bankshield.auth.context import Principal, current_principal  # noqa: E402
from bankshield.investigation import data_access  # noqa: E402
from bankshield.investigation.agent import run_investigation  # noqa: E402
from bankshield.investigation.approvals import get_approval_store  # noqa: E402
from bankshield.investigation.bedrock_diagnostics import explain_denial  # noqa: E402
from bankshield.investigation.bedrock_live import redact, run_preflight, verify_live_result  # noqa: E402
from bankshield.investigation.case_store import get_case_store  # noqa: E402
from bankshield.investigation.llm_client import BedrockClaudeClient, LLMServiceError  # noqa: E402

OUT_DIR = ROOT / "docs" / "examples"
ANALYST, REVIEWER = "live.analyst", "live.reviewer"


def _pick_tier1() -> str:
    for txn in data_access.sample_transactions(n=10):
        if txn["risk_tier"] == "tier_1":
            return txn["transaction_id"]
    raise SystemExit("no Tier 1 alert in the sample; pass --transaction-id")


def _markdown(example: dict) -> str:
    lines = [
        "# Live Amazon Bedrock investigation (sanitized)",
        "",
        f"Recorded {example['recorded_at']} by `scripts/run_live_bedrock_investigation.py`. "
        "Synthetic transaction data; AWS identifiers redacted.",
        "",
        f"- Model: `{example['model_id']}` in `{example['region']}`",
        f"- Transaction: `{example['transaction_id']}` ({example['risk_tier']}, score {example['risk_score']:.3f})",
        f"- Tokens: {example['input_tokens']} in / {example['output_tokens']} out, "
        f"est. ${example['estimated_cost_usd']:.4f}, {example['latency_ms'] / 1000:.1f}s",
        "",
        "## Checks",
        "",
        *[f"- {'PASS' if c['passed'] else 'FAIL'} **{c['name']}**: {c['detail']}" for c in example["checks"]],
        "",
        "## Tool calls",
        "",
        *[f"{i}. `{t['tool_name']}` {json.dumps(t['input'], ensure_ascii=False)}" for i, t in enumerate(example["tool_calls"], 1)],
        "",
        "## Narrative",
        "",
        example["narrative"],
        "",
        "## Verified citations",
        "",
        *[f"- [{c['doc_id']} {c['section']}] {c.get('title', '')}" for c in example["citations"]],
        "",
        "## Human-approval gate",
        "",
        *[f"- {a['action']} proposed by `{a['requested_by']}` -> **{a['status_after_agent']}** after the "
          f"agent run; `{a['reviewer']}` then approved -> case created: {a['case_created_after_review']}"
          for a in example["approvals"]],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-billable", action="store_true", help="Acknowledge this makes paid Bedrock calls")
    parser.add_argument("--transaction-id")
    args = parser.parse_args()
    if not args.confirm_billable:
        print("Refusing to run: this makes billable Amazon Bedrock calls. Re-run with --confirm-billable.")
        return 2

    report = run_preflight()
    for line in report.blockers:
        print(f"[FAIL] {line}")
    if not report.ready:
        return 1

    txn_id = args.transaction_id or _pick_tier1()
    client = BedrockClaudeClient()
    print(f"Investigating {txn_id} with {client.model_id} in {client.region} ...")
    cases_before = len(get_case_store().list_all())
    # Attribute the proposal to an analyst, so the reviewer below is a
    # different person (the four-eyes rule refuses self-approval).
    current_principal.set(Principal(ANALYST, "analyst"))
    try:
        result = run_investigation(txn_id, client)
    except LLMServiceError as exc:
        print(f"[FAIL] Bedrock call failed ({exc.kind}, {exc.aws_code or 'no code'}, request {exc.request_id}): {exc}")
        if exc.aws_message:
            print(f"       AWS says: {exc.aws_message}")
        if exc.aws_code == "AccessDeniedException" and exc.aws_message:
            e = explain_denial(exc.aws_message)
            print(f"       denied: {e.action or '?'} on {e.resource or '?'} -> {e.bucket} ({e.cause.category})")
            print(f"       fix: {e.cause.fix}")
        return 1
    cases_after = len(get_case_store().list_all())

    checks = verify_live_result(result, cases_before, cases_after)
    for c in checks:
        print(f"[{' OK ' if c.passed else 'FAIL'}] {c.name}: {c.detail}")

    approvals = []
    for proposal in result.pending_approvals:
        decided = get_approval_store().decide(proposal.approval_id, approved=True, reviewer=REVIEWER)
        created = any(c.approval_id == decided.approval_id for c in get_case_store().list_all())
        approvals.append({
            "action": proposal.action,
            "requested_by": proposal.requested_by,
            "status_after_agent": proposal.status.value,
            "reviewer": decided.reviewer,
            "case_created_after_review": created,
        })

    if not all(c.passed for c in checks):
        print("\nNot saving an example: a check failed (see above).")
        return 1

    example = redact({
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_id": result.model_id,
        "region": client.region,
        "transaction_id": txn_id,
        "risk_tier": result.payload.risk.risk_tier.value,
        "risk_score": result.payload.risk.risk_score,
        "disposition": result.disposition,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "latency_ms": round(result.latency_ms),
        "checks": [c.__dict__ for c in checks],
        "tool_calls": [{"tool_name": t.tool_name, "input": t.input, "is_error": t.is_error} for t in result.tool_calls],
        "narrative": result.narrative,
        "citations": [c.model_dump(mode="json") for c in result.citations],
        "approvals": approvals,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "live_bedrock_investigation.json").write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "live_bedrock_investigation.md").write_text(_markdown(example))
    print(f"\nSaved sanitized example to {OUT_DIR}/live_bedrock_investigation.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
