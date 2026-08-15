"""Investigation agent: the tool-use loop that turns a transaction_id into
an analyst-facing, cited explanation.

Orchestration is a manual loop (not a framework) against `LLMClient.converse`
(llm_client.py): call the model, execute any requested tools, feed results
back, repeat until the model reaches `end_turn` or a turn cap. This keeps
the loop identical whether it's driven by the real Bedrock client or
`FakeLLMClient` in tests -- both speak the same Converse-shaped messages.

Grounding: before the loop starts, `build_investigation_payload` assembles
an `InvestigationPayload` (schemas.py) from Phase 1-3 data via the same
functions the agent's own tools call (data_access.py) -- so the system
prompt's initial context and everything a tool call can subsequently fetch
are drawn from the same source of truth, not two separate paths that could
drift apart.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from bankshield import config
from bankshield.investigation import data_access, tools
from bankshield.investigation.approvals import get_approval_store
from bankshield.investigation.llm_client import ContentBlock, LLMClient, TextBlock, ToolUseBlock
from bankshield.investigation.rag import search_policy
from bankshield.investigation.schemas import (
    AuthEvent,
    CyberSignals,
    FeatureContribution,
    GraphNeighbor,
    GraphSignals,
    InvestigationPayload,
    InvestigationResult,
    PendingApproval,
    PolicyCitation,
    RiskTier,
    ToolCallRecord,
    TransactionEvidence,
    TransactionRisk,
)

SYSTEM_PROMPT = """\
You are BankShield's AI investigation assistant, helping a human fraud/AML \
analyst triage transaction alerts.

Ground every factual claim in tool output, not prior knowledge: call \
get_risk_score before describing risk, get_auth_history before describing \
login behavior, get_graph_neighbors before describing network connections, \
and search_policy before citing a policy -- cite retrieved passages as \
[DOC-ID §section], e.g. [POL-ATO-003 §2.1]. Do not invent a citation you \
did not retrieve.

Do not present a one-sided narrative: if evidence weighs against your \
recommended disposition, say so.

create_case is a consequential action. Calling it only files a proposal for \
human approval -- it never creates anything by itself. After calling it, \
report the proposal as pending, never as completed. Do not claim a case, \
account restriction, or SAR was created or filed unless a tool result says \
so explicitly.

End with a concise disposition: confirmed_fraud, false_positive, or \
inconclusive_monitor, and your reasoning, citing specific evidence."""

# Public: reused by the evaluation harness to find *every* citation-shaped
# mention in a narrative, including ones that don't resolve to a real chunk
# (i.e. potential hallucinations) -- see investigation/evaluation.py.
CITATION_PATTERN = re.compile(r"\[(POL-[A-Z0-9-]+)\s+(§[\d.]+(?: \(cont\. \d+\))?)\]")
_CITATION_RE = CITATION_PATTERN


def build_investigation_payload(transaction_id: str) -> InvestigationPayload:
    txn = data_access.get_transaction(transaction_id)
    risk = data_access.get_risk_score(transaction_id)
    customer_id = txn["customer_id"]
    auth_history = data_access.get_auth_history(customer_id, before=txn["timestamp"])
    neighbors = data_access.get_graph_neighbors(customer_id)

    top_features = [FeatureContribution(**f) for f in risk["top_contributing_features"]]
    feature_query_terms = " ".join(f.feature for f in top_features[:3])
    policy_hits = search_policy(f"{txn['merchant_category']} {feature_query_terms}")

    return InvestigationPayload(
        transaction_id=transaction_id,
        transaction=TransactionEvidence(**{k: txn[k] for k in TransactionEvidence.model_fields}),
        risk=TransactionRisk(
            transaction_id=transaction_id,
            model_version=risk["model_version"],
            risk_score=risk["risk_score"],
            risk_tier=RiskTier(risk["risk_tier"]),
            top_contributing_features=top_features,
        ),
        cyber_signals=CyberSignals(
            failed_logins_1h=txn["cyber_failed_logins_1h"],
            login_count_24h=txn["cyber_login_count_24h"],
            minutes_since_last_login=txn["cyber_minutes_since_last_login"],
            new_device_recent=txn["cyber_new_device_recent"],
            unusual_country_recent=txn["cyber_unusual_country_recent"],
            recent_suspicious_auth=txn["cyber_recent_suspicious_auth"],
        ),
        graph_signals=GraphSignals(
            shared_device_count=txn["graph_shared_device_count"],
            shared_ip_count=txn["graph_shared_ip_count"],
            beneficiary_connectivity=txn["graph_beneficiary_connectivity"],
            suspicious_neighbor_count=txn["graph_suspicious_neighbor_count"],
            account_network_risk=txn["graph_account_network_risk"],
        ),
        recent_auth_history=[AuthEvent(**e) for e in auth_history],
        graph_neighbors=[GraphNeighbor(**n) for n in neighbors],
        retrieved_policy=[PolicyCitation(**h) for h in policy_hits],
    )


def _payload_briefing(payload: InvestigationPayload) -> str:
    """A compact JSON briefing for the initial user turn -- not the full
    payload object (which includes the whole auth history / neighbor list),
    just what's needed to start reasoning; the agent can call tools for
    more detail."""
    briefing = {
        "transaction_id": payload.transaction_id,
        "transaction": payload.transaction.model_dump(mode="json"),
        "risk_score": payload.risk.risk_score,
        "risk_tier": payload.risk.risk_tier.value,
        "cyber_signals": payload.cyber_signals.model_dump(),
        "graph_signals": payload.graph_signals.model_dump(),
    }
    return (
        "Investigate this alert and produce a cited explanation for the analyst:\n\n"
        f"{json.dumps(briefing, indent=2, default=str)}"
    )


def _content_to_converse(blocks: list[ContentBlock]) -> list[dict]:
    out = []
    for block in blocks:
        if isinstance(block, TextBlock):
            out.append({"text": block.text})
        elif isinstance(block, ToolUseBlock):
            out.append({"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.input}})
    return out


def _tool_results_to_converse(results: list[tuple[ToolUseBlock, dict, bool]]) -> list[dict]:
    out = []
    for tool_use, output, is_error in results:
        out.append(
            {
                "toolResult": {
                    "toolUseId": tool_use.id,
                    "content": [{"text": json.dumps(output, default=str)}],
                    "status": "error" if is_error else "success",
                }
            }
        )
    return out


def _extract_citations(narrative: str) -> list[PolicyCitation]:
    from bankshield.investigation.rag import get_retriever

    retriever = get_retriever()
    citations = []
    seen = set()
    for doc_id, section in _CITATION_RE.findall(narrative):
        key = (doc_id, section)
        if key in seen:
            continue
        seen.add(key)
        chunk = retriever.get_by_id(doc_id, section)
        if chunk is not None:
            citations.append(
                PolicyCitation(
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    title=f"{chunk.doc_title} — {chunk.section_title}",
                    text=chunk.text,
                    score=1.0,
                )
            )
    return citations


def _estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    for key, prices in config.BEDROCK_PRICE_PER_1K_TOKENS.items():
        if key in model_id:
            return (input_tokens / 1000) * prices["input"] + (output_tokens / 1000) * prices["output"]
    return 0.0


@dataclass
class AgentRunConfig:
    max_turns: int = 8


def run_investigation(
    transaction_id: str,
    llm_client: LLMClient,
    run_config: AgentRunConfig | None = None,
) -> InvestigationResult:
    run_config = run_config or AgentRunConfig()
    start = time.monotonic()

    payload = build_investigation_payload(transaction_id)
    messages: list[dict] = [{"role": "user", "content": [{"text": _payload_briefing(payload)}]}]

    tool_call_records: list[ToolCallRecord] = []
    pending_approvals: list[PendingApproval] = []
    total_input_tokens = 0
    total_output_tokens = 0
    narrative = ""
    model_id = getattr(llm_client, "model_id", "unknown")

    for _ in range(run_config.max_turns):
        response = llm_client.converse(messages, system=SYSTEM_PROMPT, tools=tools.TOOL_SPECS)
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        model_id = response.model_id

        messages.append({"role": "assistant", "content": _content_to_converse(response.content)})

        text = response.text()
        if text:
            narrative = text if not narrative else f"{narrative}\n{text}"

        if response.stop_reason != "tool_use":
            break

        tool_uses = response.tool_uses()
        results: list[tuple[ToolUseBlock, dict, bool]] = []
        for tool_use in tool_uses:
            exec_result = tools.execute_tool(tool_use.name, tool_use.input)
            tool_call_records.append(exec_result.record)
            results.append((tool_use, exec_result.output, exec_result.record.is_error))

            if tool_use.name in tools.CONSEQUENTIAL_TOOLS and not exec_result.record.is_error:
                approval_id = exec_result.output.get("approval_id")
                if approval_id:
                    approval = get_approval_store().get(approval_id)
                    if approval is not None:
                        pending_approvals.append(approval)

        messages.append({"role": "user", "content": _tool_results_to_converse(results)})
    else:
        narrative += "\n[investigation stopped: reached max_turns without a final answer]"

    disposition = "inconclusive_monitor"
    create_case_calls = [r for r in tool_call_records if r.tool_name == "create_case" and not r.is_error]
    if create_case_calls:
        # The proposed disposition from the (not-yet-approved) case proposal
        # is the agent's stated conclusion, even though the case itself is
        # pending human approval.
        disposition = create_case_calls[-1].input.get("disposition", disposition)
    else:
        for candidate in ("confirmed_fraud", "false_positive", "inconclusive_monitor"):
            if candidate in narrative:
                disposition = candidate
                break

    citations = _extract_citations(narrative)
    latency_ms = (time.monotonic() - start) * 1000

    return InvestigationResult(
        transaction_id=transaction_id,
        payload=payload,
        narrative=narrative,
        citations=citations,
        disposition=disposition,
        tool_calls=tool_call_records,
        pending_approvals=pending_approvals,
        model_id=model_id,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        estimated_cost_usd=_estimate_cost_usd(model_id, total_input_tokens, total_output_tokens),
        latency_ms=latency_ms,
    )
