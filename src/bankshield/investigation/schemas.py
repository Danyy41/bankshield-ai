"""Structured payloads for Phase 4 AI-assisted investigation.

These Pydantic models are the contract between the data/service layer
(`data_access.py`, `tools.py`) and everything downstream that consumes it:
the LLM agent (which is grounded in an `InvestigationPayload`), the API
layer (which serializes these as JSON), and the evaluation harness (which
checks claims in generated text against the values here).

Nothing in this module touches Phase 1-3 code or data -- it only describes
shapes for data those phases already produce.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    TIER_1 = "tier_1"  # score >= 0.80, see POL-AML-001 SS2.1
    TIER_2 = "tier_2"  # 0.50 <= score < 0.80
    TIER_3 = "tier_3"  # score < 0.50


class FeatureContribution(BaseModel):
    """One feature's contribution to the model's margin for a single
    transaction, from XGBoost's native `pred_contribs` output -- an exact
    per-row attribution, not an approximation."""

    feature: str
    value: Any
    contribution: float = Field(
        description="Signed contribution to the model's raw margin. Positive pushes toward fraud."
    )


class TransactionRisk(BaseModel):
    transaction_id: str
    model_version: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_tier: RiskTier
    top_contributing_features: list[FeatureContribution]


class CyberSignals(BaseModel):
    failed_logins_1h: int
    login_count_24h: int
    minutes_since_last_login: float
    new_device_recent: bool
    unusual_country_recent: bool
    recent_suspicious_auth: bool = Field(
        description="Composite ATO-shape flag; see POL-ATO-003 SS2.1"
    )


class GraphSignals(BaseModel):
    shared_device_count: int
    shared_ip_count: int
    beneficiary_connectivity: int
    suspicious_neighbor_count: int
    account_network_risk: float = Field(ge=0.0, le=1.0)


class TransactionEvidence(BaseModel):
    """Raw transaction facts -- not derived, straight from the record."""

    transaction_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    merchant_category: str
    home_country: str
    country: str
    country_mismatch: bool
    device_id: str
    new_device: bool
    ip_address: str
    account_age_days: int
    transaction_velocity_24h: int
    new_beneficiary: bool
    is_night: bool
    amount_to_avg_ratio: float


class AuthEvent(BaseModel):
    timestamp: datetime
    device_id: str
    ip_address: str
    country: str
    login_success: bool
    new_device: bool
    new_location: bool
    session_duration_seconds: float


class GraphNeighbor(BaseModel):
    customer_id: str
    shared_via: list[str] = Field(description="e.g. ['device', 'beneficiary']")
    has_prior_fraud: bool


class PolicyCitation(BaseModel):
    doc_id: str
    section: str
    title: str
    text: str
    score: float = Field(description="Retrieval relevance score, higher is more relevant")


class InvestigationPayload(BaseModel):
    """The full structured context for one alert: everything the agent is
    grounded in before it starts reasoning or calling tools."""

    transaction_id: str
    transaction: TransactionEvidence
    risk: TransactionRisk
    cyber_signals: CyberSignals
    graph_signals: GraphSignals
    recent_auth_history: list[AuthEvent]
    graph_neighbors: list[GraphNeighbor]
    retrieved_policy: list[PolicyCitation]


class ToolCallRecord(BaseModel):
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any] = Field(default_factory=dict)
    output_summary: str
    is_error: bool = False
    latency_ms: float


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PendingApproval(BaseModel):
    approval_id: str
    transaction_id: str
    action: str = Field(description="e.g. 'create_case'")
    proposed_input: dict[str, Any]
    rationale: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime
    decided_at: datetime | None = None
    reviewer: str | None = None
    decision_notes: str | None = None


class Case(BaseModel):
    case_id: str
    transaction_id: str
    customer_id: str
    disposition: str = Field(description="confirmed_fraud | false_positive | inconclusive_monitor")
    summary: str
    cited_evidence: list[str]
    approval_id: str
    created_at: datetime


class InvestigationResult(BaseModel):
    """Top-level response for an investigation run: the grounding payload,
    the generated analyst-facing explanation, its citations, every tool
    call made along the way, and any action awaiting human approval."""

    transaction_id: str
    payload: InvestigationPayload
    narrative: str
    citations: list[PolicyCitation]
    disposition: str
    tool_calls: list[ToolCallRecord]
    pending_approvals: list[PendingApproval]
    model_id: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
