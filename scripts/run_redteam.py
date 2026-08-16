"""Phase 5: run the BankShield AI red-team suite.

Loads security/redteam_cases.yaml, executes every case's handler
(src/bankshield/security/redteam_engine.py) against the real, offline
BankShield stack (no AWS credentials, no live model -- see
redteam_engine.py's module docstring for why that's the right threat
model here), and writes reports/phase5_redteam_report.md with a full
per-case table, the overall attack-block rate, and limitations.

This suite never fabricates results: every case's PASS/FAIL comes from
actually running the corresponding handler and inspecting real system
state (approval stores, case stores, HTTP responses, citation lists) --
see run_case()/run_all() in redteam_engine.py.

Usage:
    python scripts/run_redteam.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from bankshield import config
from bankshield.security.redteam_engine import RedTeamCaseResult, run_case

CATEGORY_LABELS = {
    "prompt_injection": "1. Prompt injection",
    "rag_poisoning": "2. RAG / document poisoning",
    "unauthorized_tool_use": "3. Unauthorized tool use",
    "approval_bypass": "4. Human approval bypass",
    "citation_integrity": "5. Citation integrity",
    "information_leakage": "6. Sensitive-data / information leakage",
    "adversarial_api_input": "7. Adversarial API inputs",
    "privilege_escalation": "8. Tool argument validation / privilege escalation",
}


def load_cases() -> list[dict]:
    raw = yaml.safe_load(config.REDTEAM_CASES_YAML.read_text())
    return raw["cases"]


def run_suite(cases: list[dict]) -> list[tuple[dict, RedTeamCaseResult]]:
    results = []
    for case in cases:
        result = run_case(case["attack_id"])
        results.append((case, result))
    return results


def _fmt(text: str) -> str:
    """Collapse YAML block-scalar whitespace and escape pipes for a
    Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def build_report(results: list[tuple[dict, RedTeamCaseResult]]) -> str:
    n_total = len(results)
    n_pass = sum(1 for _, r in results if r.passed)
    block_rate = n_pass / n_total if n_total else 0.0

    lines = [
        "# BankShield AI -- Phase 5: AI Security / Red-Team Evaluation Report",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"{n_total} synthetic attack cases across 8 categories, executed against "
        "the real, offline BankShield investigation stack "
        "(`AutoFakeLLMClient`/`FakeLLMClient`, no AWS credentials, no live model).",
        "",
        "> **This is a synthetic, portfolio-scope security evaluation of this "
        "repository's own offline harness. It is not a live-model red-team and "
        "is not a claim of production security.** See Limitations below.",
        "",
        "## Overall result",
        "",
        f"**Attack-block rate: {n_pass}/{n_total} ({block_rate:.0%})**",
        "",
    ]

    by_category: dict[str, list[tuple[dict, RedTeamCaseResult]]] = {}
    for case, result in results:
        by_category.setdefault(case["category"], []).append((case, result))

    lines += [
        "| Category | Blocked | Total |",
        "|---|---|---|",
    ]
    for cat_key, label in CATEGORY_LABELS.items():
        rows = by_category.get(cat_key, [])
        if not rows:
            continue
        cat_pass = sum(1 for _, r in rows if r.passed)
        lines.append(f"| {label} | {cat_pass} | {len(rows)} |")
    lines.append("")

    lines += [
        "## Per-case results",
        "",
        "| ID | Threat | Test case | Expected behavior | Actual behavior | Result | Mitigation |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, result in results:
        verdict = "**PASS**" if result.passed else "**FAIL**"
        lines.append(
            f"| `{case['attack_id']}` | {_fmt(case['threat'])} | {_fmt(case['description'])} | "
            f"{_fmt(case['expected_security_behavior'])} | {_fmt(result.actual_behavior)} | "
            f"{verdict} | {_fmt(case['mitigation'])} |"
        )
    lines.append("")

    failing = [(c, r) for c, r in results if not r.passed]
    if failing:
        lines += ["## Open findings", ""]
        for case, result in failing:
            lines.append(f"- **{case['attack_id']}** ({case['category']}): {result.actual_behavior}")
        lines.append("")
    else:
        lines += ["## Open findings", "", "None -- every case in this suite passed on the run that produced this report.", ""]

    lines += [
        "## Attack categories tested",
        "",
        "1. **Prompt injection** -- narrative claims of already-completed actions, "
        "instruction-shaped API field values, injected enum values.",
        "2. **RAG / document poisoning** -- a synthetic malicious policy document, "
        "kept structurally isolated from the production corpus "
        "(`security/malicious_policy_fixtures/`), tested both for accidental "
        "ingestion and for whether its content -- if retrieved -- could authorize "
        "a consequential action.",
        "3. **Unauthorized tool use** -- tool names outside the allowlist, "
        "near-miss name variants, and direct (agent-loop-bypassing) tool calls.",
        "4. **Human approval bypass** -- blank reviewer, double-decision, and a "
        "forged non-approved approval object handed straight to the case store.",
        "5. **Citation integrity** -- nonexistent citations, real-but-unretrieved "
        "citations (the specific gap this phase's fix closes), and malformed "
        "citation shapes.",
        "6. **Sensitive-data / information leakage** -- direct requests for the "
        "system prompt/credentials via search_policy, injection-style customer "
        "IDs, and a simulated internal exception containing sensitive-looking text.",
        "7. **Adversarial API inputs** -- oversized IDs, malformed JSON bodies, "
        "and nonexistent object IDs across every lookup endpoint.",
        "8. **Tool argument validation / privilege escalation** -- smuggled extra "
        "fields, out-of-range numeric arguments.",
        "",
        "## Defenses implemented (src/bankshield/security/)",
        "",
        "- **`validators.py`** -- a Pydantic model per tool "
        "(`extra=\"forbid\"`, bounded lengths, enum-constrained fields e.g. "
        "`create_case`'s `disposition`), enforced server-side in "
        "`tools.execute_tool` before dispatch -- independent of whatever schema "
        "compliance the calling model claims.",
        "- **`guardrails.py`** -- `safe_execute_tool` fail-closed containment "
        "around tool execution (an unauthorized tool name becomes a contained "
        "error result, not a crash); `filter_citations_to_retrieved` / "
        "`retrieved_chunk_ids` restrict citations to what THIS run actually "
        "retrieved, not merely what exists in the corpus; "
        "`sanitize_error_response` backs a global FastAPI exception handler so "
        "no unexpected exception message ever reaches a caller.",
        "- **Untouched, pre-existing structural boundary**: `create_case` never "
        "creates a `Case` -- it only ever files a `PendingApproval`; a `Case` is "
        "constructed exclusively inside `CaseStore.create_from_approval`, "
        "reachable only from `ApprovalStore.decide(approved=True)`, itself only "
        "reachable from the human-facing `POST /approvals/{id}/decision` "
        "endpoint. Phase 5 adds one defense-in-depth assertion "
        "(`create_from_approval` now also checks `approval.status == APPROVED`) "
        "but did not need to change this boundary to pass the suite.",
        "",
        "## Limitations",
        "",
        "- **Offline only.** Every prompt-injection and RAG-poisoning case is "
        "executed against a deterministic, scripted `FakeLLMClient` -- there is "
        "no live-model evaluation here (this repo's public demo runs without AWS "
        "credentials by design; see `BANKSHIELD_LLM_MODE`). The scripts "
        "intentionally emit the tool calls / narrative text a *successfully* "
        "jailbroken or injected model would produce, and test whether the "
        "surrounding system (validators, guardrails, approval gate) still holds "
        "-- this validates the harness's resilience to a compromised model, not "
        "whether a live Claude model can actually be socially engineered by "
        "clever wording. That would require a live Bedrock evaluation, "
        "explicitly out of scope for this offline, credential-free portfolio.",
        "- **In-process, single-actor threat model.** All stores "
        "(`ApprovalStore`, `CaseStore`, the API's investigation-results cache) "
        "are in-memory and process-local, as documented since Phase 4 -- this "
        "suite does not model multi-instance races, network-level attacks (TLS, "
        "auth, rate limiting), or an attacker with direct database/filesystem "
        "access.",
        "- **No authentication/authorization layer.** This API has no user "
        "accounts or API keys (`reviewer` is a free-text field, not a verified "
        "identity) -- a production deployment would need one; this suite only "
        "verifies that *whoever* calls the API cannot bypass the "
        "approval/schema/allowlist structure, not that the caller is who they "
        "claim to be.",
        "- **24 deterministic cases, not exhaustive fuzzing.** Each case targets "
        "one concrete, previously-identified failure mode with a specific "
        "input; it is not a property-based/fuzz test suite and does not claim "
        "to cover every possible malicious input shape.",
        "- **Citation/RAG defenses are structural, not semantic.** The system "
        "does not attempt to classify document *content* as malicious (an "
        "explicit design choice, avoiding brittle keyword-based content "
        "filtering) -- it only ever restricts what can be treated as a "
        "*verified citation* (retrieved-this-run) and denies documents any "
        "authority to trigger side effects. A poisoned document that was "
        "genuinely retrieved can still influence the model's *narrative text*; "
        "it just cannot create a case, approve anything, or fabricate a citation "
        "that survives the retrieval-scoping filter.",
        "- **This is a portfolio project.** The findings and mitigations above "
        "are real (see the fixed citation-scoping gap, CITE-02), but this "
        "evaluation was performed by the same engineer who built the system, on "
        "synthetic data, without an independent red team, a bug bounty, or a "
        "formal pentest. It demonstrates security-conscious engineering "
        "practice, not a certification of production readiness.",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    cases = load_cases()
    results = run_suite(cases)

    for case, result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {case['attack_id']:8} {case['category']}")

    n_pass = sum(1 for _, r in results if r.passed)
    print(f"\n{n_pass}/{len(results)} attacks blocked ({n_pass / len(results):.0%})")

    report = build_report(results)
    config.REDTEAM_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.REDTEAM_REPORT_PATH.write_text(report)
    print(f"\nSaved report to {config.REDTEAM_REPORT_PATH}")

    failing = [c for c, r in results if not r.passed]
    if failing:
        sys.exit(1)


if __name__ == "__main__":
    main()
