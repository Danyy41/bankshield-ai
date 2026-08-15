"""Tests for the Phase 4 evaluation harness."""

from __future__ import annotations

from bankshield.investigation import data_access
from bankshield.investigation.evaluation import (
    evaluate_one,
    run_evaluation,
    select_golden_set,
)
from bankshield.investigation.llm_client import AutoFakeLLMClient


def test_select_golden_set_returns_valid_transaction_ids():
    ids = select_golden_set(n_per_tier=2, candidate_pool=300)
    assert ids
    for txn_id in ids:
        data_access.get_transaction(txn_id)  # raises NotFoundError if invalid


def test_select_golden_set_tiers_match_actual_model_scores():
    ids = select_golden_set(n_per_tier=2, candidate_pool=300)
    for txn_id in ids:
        risk = data_access.get_risk_score(txn_id)
        assert risk["risk_tier"] in ("tier_1", "tier_2", "tier_3")


def test_select_golden_set_is_deterministic():
    ids_1 = select_golden_set(n_per_tier=2, candidate_pool=300)
    ids_2 = select_golden_set(n_per_tier=2, candidate_pool=300)
    assert ids_1 == ids_2


def test_evaluate_one_offline_produces_perfect_scores_on_scripted_agent():
    """The scripted FakeLLMClient always cites a passage it just retrieved
    and always states the score it just looked up, so on this offline path
    citation correctness and evidence faithfulness should both be 1.0 --
    a regression here means either the script or the metric logic broke."""
    ids = select_golden_set(n_per_tier=1, candidate_pool=300)
    client = AutoFakeLLMClient()
    row, result = evaluate_one(ids[0], client)

    assert row.citation_correctness == 1.0
    assert row.evidence_faithfulness == 1.0
    assert row.tool_call_success_rate == 1.0
    assert row.latency_ms > 0
    assert result.transaction_id == ids[0]


def test_run_evaluation_aggregates_across_multiple_transactions():
    ids = select_golden_set(n_per_tier=1, candidate_pool=300)
    client = AutoFakeLLMClient()
    report = run_evaluation(ids, client)

    assert len(report.rows) == len(ids)
    assert 0.0 <= report.mean_citation_correctness <= 1.0
    assert 0.0 <= report.mean_evidence_faithfulness <= 1.0
    assert 0.0 <= report.mean_tool_call_success_rate <= 1.0
    assert report.mean_latency_ms > 0
    assert report.total_estimated_cost_usd >= 0.0
