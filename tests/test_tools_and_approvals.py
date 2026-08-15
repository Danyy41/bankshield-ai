"""Tests for the Phase 4 tool registry and the human-approval gate on
consequential actions (POL-CASE-004 §3)."""

from __future__ import annotations

import pytest

from bankshield.investigation import data_access, tools
from bankshield.investigation.approvals import ApprovalStore
from bankshield.investigation.case_store import CaseStore
from bankshield.investigation.schemas import ApprovalStatus


@pytest.fixture(scope="module")
def sample_transaction_id():
    store = data_access.get_store()
    return store.transactions.iloc[0]["transaction_id"]


def test_all_tool_specs_have_required_fields():
    for spec in tools.TOOL_SPECS:
        assert spec["name"]
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"
        assert "properties" in spec["input_schema"]


def test_tool_names_match_dispatch_table():
    assert tools.TOOL_NAMES == set(tools._DISPATCH.keys())


def test_get_transaction_tool_dispatch(sample_transaction_id):
    result = tools.execute_tool("get_transaction", {"transaction_id": sample_transaction_id})
    assert not result.record.is_error
    assert result.output["transaction_id"] == sample_transaction_id


def test_get_risk_score_tool_dispatch(sample_transaction_id):
    result = tools.execute_tool("get_risk_score", {"transaction_id": sample_transaction_id})
    assert not result.record.is_error
    assert "risk_score" in result.output


def test_search_policy_tool_dispatch():
    result = tools.execute_tool("search_policy", {"query": "structuring indicators"})
    assert not result.record.is_error
    assert "results" in result.output
    assert len(result.output["results"]) > 0


def test_unknown_tool_raises():
    with pytest.raises(tools.ToolExecutionError):
        tools.execute_tool("delete_all_customers", {})


def test_tool_call_with_bad_input_returns_error_record_not_exception():
    result = tools.execute_tool("get_transaction", {"transaction_id": "no-such-id"})
    assert result.record.is_error
    assert "error" in result.output


# --- The human-approval gate ------------------------------------------------


def test_create_case_tool_never_creates_a_case_directly(sample_transaction_id):
    """Calling the create_case tool must only ever produce a pending
    approval -- there is no code path from a tool call straight to a
    created Case."""
    case_store = CaseStore()
    approval_store = ApprovalStore()

    # Patch the module-level singletons the tool dispatch function reads,
    # so this test doesn't leak state into other tests via the shared
    # default stores.
    import bankshield.investigation.tools as tools_module

    original_approval_getter = tools_module.get_approval_store
    original_case_getter = tools_module.get_case_store
    tools_module.get_approval_store = lambda: approval_store
    tools_module.get_case_store = lambda: case_store
    try:
        result = tools.execute_tool(
            "create_case",
            {
                "transaction_id": sample_transaction_id,
                "disposition": "confirmed_fraud",
                "summary": "test rationale citing [POL-FRAUD-002 §4]",
            },
        )
    finally:
        tools_module.get_approval_store = original_approval_getter
        tools_module.get_case_store = original_case_getter

    assert result.output["status"] == "pending_approval"
    assert case_store.list_all() == []  # nothing created yet
    assert len(approval_store.list_pending()) == 1


def test_approval_decide_approved_creates_the_case(sample_transaction_id):
    approval_store = ApprovalStore()
    case_store = CaseStore()

    txn = data_access.get_transaction(sample_transaction_id)
    approval = approval_store.request(
        transaction_id=sample_transaction_id,
        action="create_case",
        proposed_input={
            "transaction_id": sample_transaction_id,
            "customer_id": txn["customer_id"],
            "disposition": "confirmed_fraud",
            "summary": "evidence-backed rationale",
            "cited_evidence": ["POL-ATO-003 §2.1"],
        },
        rationale="evidence-backed rationale",
        on_approve=lambda approved: case_store.create_from_approval(approved),
    )

    assert case_store.list_all() == []

    decided = approval_store.decide(approval.approval_id, approved=True, reviewer="analyst_jane")

    assert decided.status == ApprovalStatus.APPROVED
    assert decided.reviewer == "analyst_jane"
    cases = case_store.list_all()
    assert len(cases) == 1
    assert cases[0].transaction_id == sample_transaction_id
    assert cases[0].approval_id == approval.approval_id


def test_approval_decide_rejected_never_creates_a_case(sample_transaction_id):
    approval_store = ApprovalStore()
    case_store = CaseStore()

    approval = approval_store.request(
        transaction_id=sample_transaction_id,
        action="create_case",
        proposed_input={"transaction_id": sample_transaction_id, "disposition": "confirmed_fraud"},
        rationale="rationale",
        on_approve=lambda approved: case_store.create_from_approval(approved),
    )
    decided = approval_store.decide(
        approval.approval_id, approved=False, reviewer="analyst_jane", notes="insufficient evidence"
    )

    assert decided.status == ApprovalStatus.REJECTED
    assert decided.decision_notes == "insufficient evidence"
    assert case_store.list_all() == []


def test_cannot_decide_the_same_approval_twice():
    approval_store = ApprovalStore()
    approval = approval_store.request(
        transaction_id="txn1", action="create_case", proposed_input={}, rationale="r"
    )
    approval_store.decide(approval.approval_id, approved=True, reviewer="a")
    with pytest.raises(ValueError):
        approval_store.decide(approval.approval_id, approved=False, reviewer="b")


def test_decide_unknown_approval_raises_keyerror():
    approval_store = ApprovalStore()
    with pytest.raises(KeyError):
        approval_store.decide("apr_does_not_exist", approved=True, reviewer="a")
