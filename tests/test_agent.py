"""Tests for the Phase 4 investigation agent loop (agent.py), driven by the
offline FakeLLMClient so no AWS credentials are required."""

from __future__ import annotations

import pytest

from bankshield.investigation import data_access
from bankshield.investigation.agent import build_investigation_payload, run_investigation
from bankshield.investigation.case_store import get_case_store
from bankshield.investigation.llm_client import (
    FakeLLMClient,
    LLMResponse,
    TextBlock,
    ToolUseBlock,
    default_investigation_script,
)


@pytest.fixture(scope="module")
def store():
    return data_access.get_store()


@pytest.fixture
def tier1_transaction_id(store):
    candidates = store.transactions[
        (store.transactions["is_fraud"] == 1) & (store.transactions["graph_suspicious_neighbor_count"] > 0)
    ]
    if candidates.empty:
        pytest.skip("no transaction with graph_suspicious_neighbor_count > 0 in this dataset")
    for _, row in candidates.iterrows():
        risk = data_access.get_risk_score(row["transaction_id"])
        if risk["risk_tier"] == "tier_1":
            return row["transaction_id"]
    pytest.skip("no tier_1 transaction found among graph-connected fraud candidates")


@pytest.fixture
def tier3_transaction_id(store):
    for _, row in store.transactions.head(200).iterrows():
        risk = data_access.get_risk_score(row["transaction_id"])
        if risk["risk_tier"] == "tier_3":
            return row["transaction_id"]
    pytest.skip("no tier_3 transaction found in the first 200 rows")


def test_build_investigation_payload_matches_transaction_facts(tier1_transaction_id):
    payload = build_investigation_payload(tier1_transaction_id)
    txn = data_access.get_transaction(tier1_transaction_id)

    assert payload.transaction_id == tier1_transaction_id
    assert payload.transaction.amount == txn["amount"]
    assert payload.transaction.customer_id == txn["customer_id"]
    assert payload.cyber_signals.recent_suspicious_auth == txn["cyber_recent_suspicious_auth"]
    assert payload.graph_signals.suspicious_neighbor_count == txn["graph_suspicious_neighbor_count"]
    assert payload.risk.risk_tier.value == "tier_1"
    assert len(payload.risk.top_contributing_features) > 0


def test_run_investigation_end_to_end_tier1_proposes_case(tier1_transaction_id):
    client = FakeLLMClient(default_investigation_script(tier1_transaction_id))
    result = run_investigation(tier1_transaction_id, client)

    assert result.transaction_id == tier1_transaction_id
    assert result.disposition == "confirmed_fraud"
    assert len(result.pending_approvals) == 1
    assert result.pending_approvals[0].status.value == "pending"
    assert any(t.tool_name == "create_case" for t in result.tool_calls)
    assert all(not t.is_error for t in result.tool_calls)


def test_run_investigation_never_creates_a_case_directly(tier1_transaction_id):
    """A full agent run that proposes a case must not result in a Case
    existing anywhere -- only an explicit, separate approval decision can
    do that (see approvals.py / test_tools_and_approvals.py)."""
    client = FakeLLMClient(default_investigation_script(tier1_transaction_id))
    result = run_investigation(tier1_transaction_id, client)

    approval_id = result.pending_approvals[0].approval_id
    cases_referencing_this_approval = [
        c for c in get_case_store().list_all() if c.approval_id == approval_id
    ]
    assert cases_referencing_this_approval == []


def test_run_investigation_tier3_does_not_propose_a_case(tier3_transaction_id):
    client = FakeLLMClient(default_investigation_script(tier3_transaction_id))
    result = run_investigation(tier3_transaction_id, client)

    assert result.pending_approvals == []
    assert not any(t.tool_name == "create_case" for t in result.tool_calls)


def test_citations_are_filtered_to_only_actually_retrieved_chunks(tier1_transaction_id):
    """If the model's narrative cites a section that was never retrieved
    by any tool call in this run, that citation must not appear in
    result.citations -- it's exactly the ungrounded-citation failure mode
    RAG grounding exists to prevent."""

    def hallucinating_final_step(messages):
        return LLMResponse(
            stop_reason="end_turn",
            content=[
                TextBlock(
                    text="This looks suspicious per [POL-ZZZZ-999 §99.9], a document that was never "
                    "retrieved and does not exist. inconclusive_monitor."
                )
            ],
            input_tokens=10,
            output_tokens=10,
            model_id="fake-claude",
        )

    def step_1(messages):
        return LLMResponse(
            stop_reason="tool_use",
            content=[ToolUseBlock(id="c1", name="get_risk_score", input={"transaction_id": tier1_transaction_id})],
            input_tokens=10,
            output_tokens=10,
            model_id="fake-claude",
        )

    client = FakeLLMClient([step_1, hallucinating_final_step])
    result = run_investigation(tier1_transaction_id, client)

    assert result.citations == []  # the fabricated citation must not survive extraction


def test_run_investigation_reports_tokens_and_cost(tier3_transaction_id):
    client = FakeLLMClient(default_investigation_script(tier3_transaction_id))
    result = run_investigation(tier3_transaction_id, client)

    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms > 0
    assert result.estimated_cost_usd >= 0.0
