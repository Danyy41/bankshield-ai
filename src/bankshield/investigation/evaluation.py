"""Evaluation harness for the Phase 4 investigation agent.

Computes five things over a golden set of investigation scenarios, as
requested for Phase 4: citation correctness, evidence faithfulness,
tool-call success rate, latency, and estimated cost. Runs entirely offline
by default (via `AutoFakeLLMClient`) so it's part of the reproducible test
suite -- see `scripts/13_evaluate_investigations.py` for producing a report,
and pass a `BedrockClaudeClient` instead to evaluate the real model.

Metric definitions
-------------------
- **Citation correctness**: of every `[DOC-ID §section]`-shaped citation
  the narrative contains, what fraction (a) resolve to a real chunk in the
  policy corpus, *and* (b) were actually surfaced by retrieval during this
  investigation (either the automatic payload-building search, or an
  explicit `search_policy` tool call) -- not just present in the corpus
  somewhere. A citation to a real section the agent never retrieved is
  still counted as incorrect: it indicates an ungrounded (memorized rather
  than retrieved) citation, which is the failure mode RAG grounding is
  supposed to prevent.
- **Evidence faithfulness**: whether numeric/boolean claims the narrative
  makes about the transaction's risk score match the actual payload values
  (within a small tolerance for the score). A narrative with no checkable
  claims is vacuously faithful (nothing to contradict).
- **Tool-call success rate**: fraction of tool calls in the run that did
  not return an error.
- **Latency**: wall-clock time for the full investigation (agent loop +
  tool execution), in milliseconds.
- **Estimated cost**: `investigation.agent._estimate_cost_usd`'s list-price
  estimate for the tokens used, in USD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bankshield.investigation import data_access
from bankshield.investigation.agent import CITATION_PATTERN, AgentRunConfig, run_investigation
from bankshield.investigation.llm_client import LLMClient
from bankshield.investigation.rag import get_retriever
from bankshield.investigation.schemas import InvestigationResult

_SCORE_CLAIM_RE = re.compile(r"scored\s+(0?\.\d+)")


def select_golden_set(n_per_tier: int = 4, candidate_pool: int = 600) -> list[str]:
    """Deterministically pick transaction_ids spread across all three risk
    tiers, so the eval set exercises Tier 1 (case-proposal path), Tier 2,
    and Tier 3 (no consequential action) alike. Scores a bounded,
    deterministic candidate pool (every Nth transaction_id in sorted order)
    rather than the full dataset, to keep the harness fast.
    """
    store = data_access.get_store()
    sorted_ids = sorted(store.transactions["transaction_id"].tolist())
    step = max(1, len(sorted_ids) // candidate_pool)
    candidates = sorted_ids[::step][:candidate_pool]

    by_tier: dict[str, list[str]] = {"tier_1": [], "tier_2": [], "tier_3": []}
    for txn_id in candidates:
        if all(len(v) >= n_per_tier for v in by_tier.values()):
            break
        risk = data_access.get_risk_score(txn_id)
        tier = risk["risk_tier"]
        if len(by_tier[tier]) < n_per_tier:
            by_tier[tier].append(txn_id)

    return by_tier["tier_1"] + by_tier["tier_2"] + by_tier["tier_3"]


@dataclass
class InvestigationEvalRow:
    transaction_id: str
    risk_tier: str
    citation_correctness: float
    evidence_faithfulness: float
    tool_call_success_rate: float
    n_tool_calls: int
    n_citations_claimed: int
    latency_ms: float
    estimated_cost_usd: float
    disposition: str


def _retrieved_chunk_ids(result: InvestigationResult) -> set[str]:
    ids = {f"{c.doc_id} {c.section}" for c in result.payload.retrieved_policy}
    for call in result.tool_calls:
        if call.tool_name == "search_policy" and not call.is_error:
            for hit in call.output.get("results", []):
                ids.add(f"{hit['doc_id']} {hit['section']}")
    return ids


def _citation_correctness(result: InvestigationResult) -> tuple[float, int]:
    mentions = CITATION_PATTERN.findall(result.narrative)
    if not mentions:
        return 1.0, 0  # vacuously correct: no citations claimed, none to be wrong

    retriever = get_retriever()
    retrieved_ids = _retrieved_chunk_ids(result)
    correct = 0
    for doc_id, section in mentions:
        chunk_id = f"{doc_id} {section}"
        exists = retriever.get_by_id(doc_id, section) is not None
        if exists and chunk_id in retrieved_ids:
            correct += 1
    return correct / len(mentions), len(mentions)


def _evidence_faithfulness(result: InvestigationResult) -> float:
    claims_checked = 0
    claims_correct = 0

    for claimed_score_str in _SCORE_CLAIM_RE.findall(result.narrative):
        claims_checked += 1
        if abs(float(claimed_score_str) - result.payload.risk.risk_score) <= 0.02:
            claims_correct += 1

    if "recent_suspicious_auth" in result.narrative or "suspicious auth" in result.narrative.lower():
        claims_checked += 1
        # The claim is faithful if the narrative's tone (mentioning ATO/suspicious
        # auth as relevant) agrees with the payload actually showing that signal.
        if result.payload.cyber_signals.recent_suspicious_auth:
            claims_correct += 1

    if claims_checked == 0:
        return 1.0
    return claims_correct / claims_checked


def _tool_call_success_rate(result: InvestigationResult) -> float:
    if not result.tool_calls:
        return 1.0
    ok = sum(1 for c in result.tool_calls if not c.is_error)
    return ok / len(result.tool_calls)


def evaluate_one(transaction_id: str, llm_client: LLMClient, run_config: AgentRunConfig | None = None) -> tuple[InvestigationEvalRow, InvestigationResult]:
    result = run_investigation(transaction_id, llm_client, run_config)
    citation_correctness, n_citations = _citation_correctness(result)
    return (
        InvestigationEvalRow(
            transaction_id=transaction_id,
            risk_tier=result.payload.risk.risk_tier.value,
            citation_correctness=citation_correctness,
            evidence_faithfulness=_evidence_faithfulness(result),
            tool_call_success_rate=_tool_call_success_rate(result),
            n_tool_calls=len(result.tool_calls),
            n_citations_claimed=n_citations,
            latency_ms=result.latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
            disposition=result.disposition,
        ),
        result,
    )


@dataclass
class EvalReport:
    rows: list[InvestigationEvalRow] = field(default_factory=list)

    @property
    def mean_citation_correctness(self) -> float:
        return _mean(r.citation_correctness for r in self.rows)

    @property
    def mean_evidence_faithfulness(self) -> float:
        return _mean(r.evidence_faithfulness for r in self.rows)

    @property
    def mean_tool_call_success_rate(self) -> float:
        return _mean(r.tool_call_success_rate for r in self.rows)

    @property
    def mean_latency_ms(self) -> float:
        return _mean(r.latency_ms for r in self.rows)

    @property
    def total_estimated_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self.rows)

    @property
    def mean_estimated_cost_usd(self) -> float:
        return _mean(r.estimated_cost_usd for r in self.rows)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_evaluation(
    transaction_ids: list[str], llm_client: LLMClient, run_config: AgentRunConfig | None = None
) -> EvalReport:
    report = EvalReport()
    for txn_id in transaction_ids:
        row, _ = evaluate_one(txn_id, llm_client, run_config)
        report.rows.append(row)
    return report
