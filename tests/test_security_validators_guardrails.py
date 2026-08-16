"""Unit-level security regression tests for the Phase 5 defensive layer
(src/bankshield/security/) -- independent of the red-team YAML suite in
test_security_redteam.py, these exercise validators.py and guardrails.py
directly."""

from __future__ import annotations

import re

import pytest

from bankshield.investigation import tools
from bankshield.investigation.schemas import PolicyCitation
from bankshield.security import guardrails, validators


# --- validators.py ------------------------------------------------------


def test_validate_tool_input_rejects_unregistered_tool_name():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("not_a_real_tool", {})


def test_validate_tool_input_rejects_non_dict_input():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("get_transaction", "not a dict")  # type: ignore[arg-type]


def test_validate_tool_input_rejects_extra_fields():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("get_transaction", {"transaction_id": "t1", "extra_field": "x"})


def test_validate_tool_input_rejects_disposition_outside_enum():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input(
            "create_case", {"transaction_id": "t1", "disposition": "not_a_real_disposition", "summary": "s"}
        )


def test_validate_tool_input_rejects_empty_required_string():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("get_transaction", {"transaction_id": ""})


def test_validate_tool_input_rejects_oversized_string():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("get_transaction", {"transaction_id": "x" * 10_000})


def test_validate_tool_input_rejects_out_of_range_limit():
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("get_auth_history", {"customer_id": "c1", "limit": 0})
    with pytest.raises(validators.ToolInputValidationError):
        validators.validate_tool_input("get_auth_history", {"customer_id": "c1", "limit": 500})


def test_validate_tool_input_accepts_valid_input_and_applies_defaults():
    validated = validators.validate_tool_input("get_auth_history", {"customer_id": "c1"})
    assert validated == {"customer_id": "c1", "limit": 20}


def test_validate_tool_input_accepts_valid_create_case_input():
    validated = validators.validate_tool_input(
        "create_case",
        {"transaction_id": "t1", "disposition": "confirmed_fraud", "summary": "s"},
    )
    assert validated["disposition"] == "confirmed_fraud"
    assert validated["cited_evidence"] == []


def test_every_tool_in_tools_module_has_a_registered_validator():
    """Structural guard: TOOL_INPUT_MODELS must cover every dispatchable
    tool -- a new tool added to tools.py without a matching entry here
    would execute with no schema enforcement at all."""
    assert set(validators.TOOL_INPUT_MODELS.keys()) == tools.TOOL_NAMES


# --- guardrails.py: safe_execute_tool ------------------------------------


def test_safe_execute_tool_contains_unknown_tool_name():
    result = guardrails.safe_execute_tool("not_a_real_tool", {})
    assert result.record.is_error
    assert "not_a_real_tool" in result.output["error"]


def test_safe_execute_tool_passes_through_successful_calls(monkeypatch):
    from bankshield.investigation import data_access

    store = data_access.get_store()
    txn_id = store.transactions.iloc[0]["transaction_id"]
    result = guardrails.safe_execute_tool("get_transaction", {"transaction_id": txn_id})
    assert not result.record.is_error
    assert result.output["transaction_id"] == txn_id


# --- guardrails.py: citation filtering -----------------------------------


class _FakeChunk:
    def __init__(self, doc_id, section, doc_title="Doc", section_title="Section", text="text"):
        self.doc_id = doc_id
        self.section = section
        self.doc_title = doc_title
        self.section_title = section_title
        self.text = text


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = {(c.doc_id, c.section): c for c in chunks}

    def get_by_id(self, doc_id, section):
        return self._chunks.get((doc_id, section))


_CITATION_RE = re.compile(r"\[(POL-[A-Z0-9-]+)\s+(§[\d.]+(?: \(cont\. \d+\))?)\]")


def test_filter_citations_to_retrieved_drops_nonexistent_citation():
    retriever = _FakeRetriever([])
    citations = guardrails.filter_citations_to_retrieved(
        "See [POL-XXX-000 §1].", _CITATION_RE, retriever, retrieved_ids=set()
    )
    assert citations == []


def test_filter_citations_to_retrieved_drops_real_but_unretrieved_citation():
    retriever = _FakeRetriever([_FakeChunk("POL-XXX-000", "§1")])
    citations = guardrails.filter_citations_to_retrieved(
        "See [POL-XXX-000 §1].", _CITATION_RE, retriever, retrieved_ids=set()  # never retrieved
    )
    assert citations == []


def test_filter_citations_to_retrieved_keeps_real_and_retrieved_citation():
    retriever = _FakeRetriever([_FakeChunk("POL-XXX-000", "§1")])
    citations = guardrails.filter_citations_to_retrieved(
        "See [POL-XXX-000 §1].", _CITATION_RE, retriever, retrieved_ids={"POL-XXX-000 §1"}
    )
    assert len(citations) == 1
    assert isinstance(citations[0], PolicyCitation)
    assert citations[0].doc_id == "POL-XXX-000"


def test_retrieved_chunk_ids_includes_payload_and_search_policy_calls():
    from bankshield.investigation.schemas import (
        AuthEvent,
        CyberSignals,
        GraphSignals,
        InvestigationPayload,
        PolicyCitation as PC,
        RiskTier,
        ToolCallRecord,
        TransactionEvidence,
        TransactionRisk,
    )
    from datetime import datetime, timezone

    payload = InvestigationPayload(
        transaction_id="t1",
        transaction=TransactionEvidence(
            transaction_id="t1", customer_id="c1", timestamp=datetime.now(timezone.utc), amount=1.0,
            merchant_category="grocery", home_country="US", country="US", country_mismatch=False,
            device_id="d1", new_device=False, ip_address="1.2.3.4", account_age_days=100,
            transaction_velocity_24h=1, new_beneficiary=False, is_night=False, amount_to_avg_ratio=1.0,
        ),
        risk=TransactionRisk(transaction_id="t1", model_version="v1", risk_score=0.1, risk_tier=RiskTier.TIER_3, top_contributing_features=[]),
        cyber_signals=CyberSignals(failed_logins_1h=0, login_count_24h=0, minutes_since_last_login=0.0, new_device_recent=False, unusual_country_recent=False, recent_suspicious_auth=False),
        graph_signals=GraphSignals(shared_device_count=0, shared_ip_count=0, beneficiary_connectivity=0, suspicious_neighbor_count=0, account_network_risk=0.0),
        recent_auth_history=[],
        graph_neighbors=[],
        retrieved_policy=[PC(doc_id="POL-A", section="§1", title="t", text="x", score=1.0)],
    )
    tool_calls = [
        ToolCallRecord(
            tool_name="search_policy",
            input={"query": "q"},
            output={"results": [{"doc_id": "POL-B", "section": "§2"}]},
            output_summary="ok",
            is_error=False,
            latency_ms=1.0,
        )
    ]
    ids = guardrails.retrieved_chunk_ids(payload, tool_calls)
    assert ids == {"POL-A §1", "POL-B §2"}


# --- guardrails.py: sanitize_error_response -------------------------------


def test_sanitize_error_response_never_includes_original_message():
    exc = RuntimeError("internal path /etc/shadow and secret token abc123")
    response = guardrails.sanitize_error_response(exc)
    assert response == {"detail": "internal error processing request"}
    assert "shadow" not in str(response)
    assert "abc123" not in str(response)
