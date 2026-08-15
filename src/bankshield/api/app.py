"""FastAPI service layer.

Two jobs, layered:

1. Expose Phase 1-3 outputs (raw transactions, model risk scores, cyber
   signals, graph neighbors, policy search) as a clean REST API -- thin
   wrappers around `investigation.data_access` / `investigation.rag`, no
   new logic.
2. Expose the Phase 4 investigation workflow: run an agent investigation,
   list/decide pending approvals, list resulting cases.

The LLM client is a FastAPI dependency (`get_llm_client`) rather than a
module-level singleton specifically so tests can override it with
`FakeLLMClient` via `app.dependency_overrides` and exercise the full API
without AWS credentials -- see tests/test_api.py.

Which backend `get_llm_client` returns is controlled by the
`BANKSHIELD_LLM_MODE` environment variable (config.LLM_MODE_ENV_VAR):
"offline" (default) -> AutoFakeLLMClient, "bedrock" -> BedrockClaudeClient.
See the README's Phase 4 section for deployment commands.
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from bankshield import config
from bankshield.investigation import data_access
from bankshield.investigation.agent import run_investigation
from bankshield.investigation.approvals import get_approval_store
from bankshield.investigation.case_store import get_case_store
from bankshield.investigation.llm_client import AutoFakeLLMClient, BedrockClaudeClient, LLMClient
from bankshield.investigation.rag import search_policy
from bankshield.investigation.schemas import Case, InvestigationResult, PendingApproval

app = FastAPI(
    title="BankShield AI — Investigation API",
    description=(
        "Phase 4: exposes Phase 1-3 fraud/cyber/graph intelligence outputs "
        "and an AI-assisted investigation workflow with a human-approval gate "
        "on consequential actions."
    ),
    version="4.0.0",
)

# transaction_id -> InvestigationResult, most-recent run wins. In-memory,
# process-local -- same tradeoff as approvals.py / case_store.py, adequate
# for a portfolio deployment.
_investigation_results: dict[str, InvestigationResult] = {}


@lru_cache(maxsize=2)
def _llm_client_for_mode(mode: str) -> LLMClient:
    """One client instance per mode, reused across requests (a fresh
    BedrockClaudeClient -- and its boto3 client -- per request would be
    wasteful). Keyed by mode so a mid-process env var change (e.g. in
    tests) still resolves to the right backend rather than a stale cache.
    """
    if mode == config.LLM_MODE_OFFLINE:
        return AutoFakeLLMClient()
    if mode == config.LLM_MODE_BEDROCK:
        return BedrockClaudeClient()
    raise RuntimeError(
        f"Invalid {config.LLM_MODE_ENV_VAR}={mode!r}; expected "
        f"{config.LLM_MODE_OFFLINE!r} or {config.LLM_MODE_BEDROCK!r}."
    )


def get_llm_client() -> LLMClient:
    """Default dependency, selected by the BANKSHIELD_LLM_MODE environment
    variable -- "offline" (default, AutoFakeLLMClient) or "bedrock"
    (BedrockClaudeClient). Overridden directly in tests via
    `app.dependency_overrides` regardless of this env var."""
    mode = os.environ.get(config.LLM_MODE_ENV_VAR, config.LLM_MODE_DEFAULT)
    return _llm_client_for_mode(mode)


# --- Phase 1-3 passthrough endpoints ----------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str) -> dict:
    try:
        return data_access.get_transaction(transaction_id)
    except data_access.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/transactions/{transaction_id}/risk")
def get_risk_score(transaction_id: str) -> dict:
    try:
        return data_access.get_risk_score(transaction_id)
    except data_access.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/customers/{customer_id}/auth-history")
def get_auth_history(customer_id: str, limit: int = 20) -> dict:
    return {"customer_id": customer_id, "events": data_access.get_auth_history(customer_id, limit=limit)}


@app.get("/customers/{customer_id}/graph-neighbors")
def get_graph_neighbors(customer_id: str) -> dict:
    return {"customer_id": customer_id, "neighbors": data_access.get_graph_neighbors(customer_id)}


class PolicySearchRequest(BaseModel):
    query: str
    top_k: int = 4


@app.post("/policy/search")
def post_policy_search(request: PolicySearchRequest) -> dict:
    return {"results": search_policy(request.query, top_k=request.top_k)}


@app.get("/demo/sample-transactions")
def get_sample_transactions(n: int = 5) -> dict:
    """Valid, deployed transaction_ids with their risk score/tier, so a
    caller with no prior knowledge of this dataset can immediately try
    `/transactions/{id}`, `/investigations`, etc. against something real --
    useful for the public demo deployment (see BANKSHIELD_LLM_MODE above)."""
    return {"transactions": list(data_access.sample_transactions(n=n))}


# --- Phase 4 investigation workflow -----------------------------------------


class InvestigationRequest(BaseModel):
    transaction_id: str


@app.post("/investigations", response_model=InvestigationResult)
def post_investigation(
    request: InvestigationRequest, llm_client: LLMClient = Depends(get_llm_client)
) -> InvestigationResult:
    try:
        result = run_investigation(request.transaction_id, llm_client)
    except data_access.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _investigation_results[request.transaction_id] = result
    return result


@app.get("/investigations/{transaction_id}", response_model=InvestigationResult)
def get_investigation(transaction_id: str) -> InvestigationResult:
    result = _investigation_results.get(transaction_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"no investigation has been run yet for transaction_id {transaction_id!r}",
        )
    return result


@app.get("/approvals", response_model=list[PendingApproval])
def list_approvals() -> list[PendingApproval]:
    return get_approval_store().list_pending()


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    reviewer: str
    notes: str | None = None


@app.post("/approvals/{approval_id}/decision", response_model=PendingApproval)
def decide_approval(approval_id: str, request: ApprovalDecisionRequest) -> PendingApproval:
    try:
        return get_approval_store().decide(
            approval_id, approved=request.approved, reviewer=request.reviewer, notes=request.notes
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/cases", response_model=list[Case])
def list_cases() -> list[Case]:
    return get_case_store().list_all()


@app.get("/cases/{case_id}", response_model=Case)
def get_case(case_id: str) -> Case:
    case = get_case_store().get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"no case with id {case_id!r}")
    return case
