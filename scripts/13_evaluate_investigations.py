"""Step 13 (Phase 4): evaluate the AI-assisted investigation agent.

Runs the full agent loop (build payload -> tool-use loop -> cited
narrative) over a golden set of transactions spanning all three risk
tiers, and reports citation correctness, evidence faithfulness, tool-call
success rate, latency, and estimated cost.

Runs offline by default (AutoFakeLLMClient, no AWS credentials needed) so
this is reproducible in CI. Pass --live to evaluate the real Bedrock/Claude
backend instead (requires AWS credentials with bedrock:InvokeModel).

Usage:
    python scripts/13_evaluate_investigations.py [--live] [--n-per-tier N]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankshield import config
from bankshield.investigation.agent import AgentRunConfig
from bankshield.investigation.evaluation import run_evaluation, select_golden_set
from bankshield.investigation.llm_client import AutoFakeLLMClient, BedrockClaudeClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use the real Bedrock/Claude backend")
    parser.add_argument("--n-per-tier", type=int, default=4)
    args = parser.parse_args()

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Selecting golden set ({args.n_per_tier} per risk tier)...")
    transaction_ids = select_golden_set(n_per_tier=args.n_per_tier)
    print(f"Selected {len(transaction_ids)} transactions.")

    llm_client = BedrockClaudeClient() if args.live else AutoFakeLLMClient()
    mode = "live (Bedrock)" if args.live else "offline (AutoFakeLLMClient)"
    print(f"Running investigations in {mode} mode...")

    report = run_evaluation(transaction_ids, llm_client, AgentRunConfig())

    metrics = {
        "mode": mode,
        "n_investigations": len(report.rows),
        "citation_correctness": report.mean_citation_correctness,
        "evidence_faithfulness": report.mean_evidence_faithfulness,
        "tool_call_success_rate": report.mean_tool_call_success_rate,
        "mean_latency_ms": report.mean_latency_ms,
        "mean_estimated_cost_usd": report.mean_estimated_cost_usd,
        "total_estimated_cost_usd": report.total_estimated_cost_usd,
        "rows": [vars(r) for r in report.rows],
    }
    metrics_path = config.METRICS_DIR / "phase4_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    lines = [
        "# BankShield AI -- Phase 4: Investigation Agent Evaluation",
        "",
        f"Mode: **{mode}**. {len(report.rows)} investigations across Tier 1/2/3 alerts "
        "(see POL-AML-001 SS2.1 for tier definitions).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Citation correctness | {report.mean_citation_correctness:.3f} |",
        f"| Evidence faithfulness | {report.mean_evidence_faithfulness:.3f} |",
        f"| Tool-call success rate | {report.mean_tool_call_success_rate:.3f} |",
        f"| Mean latency | {report.mean_latency_ms:.1f} ms |",
        f"| Mean estimated cost | ${report.mean_estimated_cost_usd:.5f} |",
        f"| Total estimated cost | ${report.total_estimated_cost_usd:.5f} |",
        "",
        "## Per-transaction results",
        "",
        "| Transaction | Tier | Citation correctness | Evidence faithfulness | "
        "Tool success | Tool calls | Citations claimed | Disposition |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        lines.append(
            f"| `{r.transaction_id[:12]}...` | {r.risk_tier} | {r.citation_correctness:.2f} | "
            f"{r.evidence_faithfulness:.2f} | {r.tool_call_success_rate:.2f} | {r.n_tool_calls} | "
            f"{r.n_citations_claimed} | {r.disposition} |"
        )
    lines += [
        "",
        "## Metric definitions",
        "",
        "- **Citation correctness** -- of every `[DOC-ID §section]` citation the "
        "narrative makes, the fraction that both exist in the policy corpus and "
        "were actually retrieved during the investigation (payload-building "
        "search or an explicit `search_policy` call) -- a citation to a real "
        "section the agent never retrieved still counts as incorrect (ungrounded).",
        "- **Evidence faithfulness** -- whether numeric/boolean claims the "
        "narrative makes (e.g. the stated risk score, ATO-pattern claims) match "
        "the actual payload values.",
        "- **Tool-call success rate** -- fraction of tool calls that did not "
        "return an error.",
        "- **Latency** -- wall-clock time for the full agent loop, including "
        "tool execution.",
        "- **Estimated cost** -- list-price token cost estimate "
        "(`config.BEDROCK_PRICE_PER_1K_TOKENS`); 0 for the offline fake client, "
        "which reports 0 tokens.",
        "",
    ]
    report_path = config.REPORTS_DIR / "phase4_eval.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
