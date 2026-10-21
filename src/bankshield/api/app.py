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

CORS is open (allow_origins=["*"]) so the static frontend (frontend/,
served from a different origin -- a laptop's file://, a static host, a
different port) can call this API from the browser. This is transport
plumbing, not business logic: every non-public endpoint requires a bearer
token, so an origin alone grants nothing.

Authentication and authorization (auth/): `POST /auth/token` exchanges
username + password for a short-lived JWT; every other endpoint except
`/health` (and the public A2A agent cards) requires one, with the role
checks in auth/dependencies.py. Approval decisions are reviewer-only and
attributed to the authenticated reviewer -- never to a name in the request
body -- and a reviewer can't decide a proposal they requested themselves.
Approvals, cases, investigation results, users and the audit log are
durable (persistence/).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from bankshield import config
from bankshield.agents.server import A2A_RPC_PATHS, mount_a2a_agents
from bankshield.auth import tokens
from bankshield.auth.context import Principal, current_principal, current_token
from bankshield.auth.dependencies import (
    CALL_AGENTS,
    DECIDE_APPROVAL,
    READ_AUDIT,
    READ_CASEWORK,
    READ_DATA,
    RUN_INVESTIGATION,
    authenticate_token,
    deny,
    login_throttle,
    require_roles,
)
from bankshield.auth.users import UserStore
from bankshield.investigation import data_access
from bankshield.investigation.agent import run_investigation
from bankshield.investigation.approvals import SelfApprovalError, get_approval_store
from bankshield.investigation.case_store import get_case_store
from bankshield.investigation.llm_client import AutoFakeLLMClient, BedrockClaudeClient, LLMClient, LLMServiceError
from bankshield.investigation.rag import search_policy
from bankshield.investigation.schemas import Case, InvestigationResult, PendingApproval
from bankshield.persistence import audit
from bankshield.persistence.db import get_database
from bankshield.persistence.investigations import InvestigationResultStore
from bankshield.security import validators
from bankshield.security.guardrails import sanitize_error_response

app = FastAPI(
    title="BankShield AI — Investigation API",
    description=(
        "Phase 4: exposes Phase 1-3 fraud/cyber/graph intelligence outputs "
        "and an AI-assisted investigation workflow with a human-approval gate "
        "on consequential actions."
    ),
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fail-closed catch-all: anything not already turned into an
    HTTPException by a route becomes a generic 500, never the raw
    exception message (which could contain file paths, internal
    identifiers, or other implementation detail). See
    security/guardrails.py:sanitize_error_response. FastAPI's own
    HTTPException handling still takes precedence over this."""
    return JSONResponse(status_code=500, content=sanitize_error_response(exc))


def _investigation_results() -> InvestigationResultStore:
    """Durable, most-recent run per transaction_id (persistence/investigations.py)."""
    return InvestigationResultStore()


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


# --- Authentication -----------------------------------------------------------


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


@app.post("/auth/token", response_model=TokenResponse)
def issue_token(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """OAuth2 password flow: username + password -> short-lived bearer token."""
    username = form.username[: validators.MAX_ID_LEN]
    if login_throttle.locked(username):
        audit.record_event(get_database(), action="auth.login", outcome="locked", actor=username)
        raise HTTPException(429, "too many failed attempts; try again later")
    user = UserStore().authenticate(username, form.password)
    if user is None:
        login_throttle.failed(username)
        audit.record_event(get_database(), action="auth.login", outcome="failed", actor=username)
        raise HTTPException(401, "incorrect username or password", {"WWW-Authenticate": "Bearer"})
    login_throttle.succeeded(username)
    audit.record_event(get_database(), action="auth.login", outcome="ok", actor=user.username)
    return TokenResponse(
        access_token=tokens.issue(user), expires_in=int(tokens.ttl().total_seconds()), role=user.role
    )


@app.get("/auth/me")
def whoami(principal: Principal = Depends(require_roles(*READ_DATA))) -> dict:
    return {"username": principal.username, "role": principal.role}


@app.get("/audit")
def get_audit_log(limit: int = 100, principal: Principal = Depends(require_roles(*READ_AUDIT))) -> list[dict]:
    return audit.recent(get_database(), limit=max(1, min(limit, 1000)))


def _id_path() -> Any:
    """A fresh bounded-length Path(...) per parameter -- FastAPI's Path()
    return value is not safe to share as a single instance across multiple
    differently-named path parameters, so each call site gets its own."""
    return Path(min_length=1, max_length=validators.MAX_ID_LEN)


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str = _id_path(), _: Principal = Depends(require_roles(*READ_DATA))) -> dict:
    try:
        return data_access.get_transaction(transaction_id)
    except data_access.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/transactions/{transaction_id}/risk")
def get_risk_score(transaction_id: str = _id_path(), _: Principal = Depends(require_roles(*READ_DATA))) -> dict:
    try:
        return data_access.get_risk_score(transaction_id)
    except data_access.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/customers/{customer_id}/auth-history")
def get_auth_history(
    customer_id: str = _id_path(), limit: int = 20, _: Principal = Depends(require_roles(*READ_DATA))
) -> dict:
    return {"customer_id": customer_id, "events": data_access.get_auth_history(customer_id, limit=limit)}


@app.get("/customers/{customer_id}/graph-neighbors")
def get_graph_neighbors(customer_id: str = _id_path(), _: Principal = Depends(require_roles(*READ_DATA))) -> dict:
    return {"customer_id": customer_id, "neighbors": data_access.get_graph_neighbors(customer_id)}


class PolicySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=validators.MAX_QUERY_LEN)
    top_k: int = Field(default=4, ge=1, le=20)


@app.post("/policy/search")
def post_policy_search(request: PolicySearchRequest, _: Principal = Depends(require_roles(*READ_DATA))) -> dict:
    return {"results": search_policy(request.query, top_k=request.top_k)}


@app.get("/demo/sample-transactions")
def get_sample_transactions(n: int = 5, _: Principal = Depends(require_roles(*READ_DATA))) -> dict:
    """Valid, deployed transaction_ids with their risk score/tier, so a
    caller with no prior knowledge of this dataset can immediately try
    `/transactions/{id}`, `/investigations`, etc. against something real --
    useful for the public demo deployment (see BANKSHIELD_LLM_MODE above)."""
    return {"transactions": list(data_access.sample_transactions(n=n))}


# --- Phase 4 investigation workflow -----------------------------------------


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=validators.MAX_ID_LEN)


@app.post("/investigations", response_model=InvestigationResult)
def post_investigation(
    request: InvestigationRequest,
    llm_client: LLMClient = Depends(get_llm_client),
    principal: Principal = Depends(require_roles(*RUN_INVESTIGATION)),
) -> InvestigationResult:
    try:
        result = run_investigation(request.transaction_id, llm_client)
    except data_access.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMServiceError as exc:
        raise _llm_unavailable(exc) from None
    _investigation_results().save(result, requested_by=principal.username)
    return result


def _llm_unavailable(exc: LLMServiceError) -> HTTPException:
    """Map a model-backend failure to a caller-safe response: 503 (with
    Retry-After) when retrying may help, 502 when the backend rejected or
    can't serve the request as configured. Nothing is saved for the run;
    the detail carries only the safe message and category, never AWS text."""
    detail = {
        "message": f"Investigation not completed: {exc}",
        "error": exc.kind,
        "retryable": exc.retryable,
        "pending_approval_ids": exc.pending_approval_ids,
    }
    if exc.retryable:
        return HTTPException(status_code=503, detail=detail, headers={"Retry-After": "30"})
    return HTTPException(status_code=502, detail=detail)


@app.get("/investigations/{transaction_id}", response_model=InvestigationResult)
def get_investigation(
    transaction_id: str = _id_path(), _: Principal = Depends(require_roles(*READ_CASEWORK))
) -> InvestigationResult:
    result = _investigation_results().get(transaction_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"no investigation has been run yet for transaction_id {transaction_id!r}",
        )
    return result


@app.get("/approvals", response_model=list[PendingApproval])
def list_approvals(_: Principal = Depends(require_roles(*READ_CASEWORK))) -> list[PendingApproval]:
    return get_approval_store().list_pending()


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool
    # The accountable reviewer is the authenticated principal. The field is
    # optional and kept for compatibility; if present it must be non-empty
    # and must name the caller -- it can never attribute a decision to
    # someone else.
    reviewer: str | None = Field(default=None, min_length=1, max_length=validators.MAX_REVIEWER_LEN)
    notes: str | None = Field(default=None, max_length=validators.MAX_NOTES_LEN)


@app.post("/approvals/{approval_id}/decision", response_model=PendingApproval)
def decide_approval(
    request: ApprovalDecisionRequest,
    approval_id: str = _id_path(),
    principal: Principal = Depends(require_roles(*DECIDE_APPROVAL)),
) -> PendingApproval:
    if request.reviewer is not None and request.reviewer != principal.username:
        deny(principal, f"impersonation of reviewer {request.reviewer!r} on {approval_id}")
        raise HTTPException(status_code=403, detail="a decision can only be made in your own name")
    try:
        return get_approval_store().decide(
            approval_id, approved=request.approved, reviewer=principal.username, notes=request.notes
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SelfApprovalError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/cases", response_model=list[Case])
def list_cases(_: Principal = Depends(require_roles(*READ_CASEWORK))) -> list[Case]:
    return get_case_store().list_all()


@app.get("/cases/{case_id}", response_model=Case)
def get_case(case_id: str = _id_path(), _: Principal = Depends(require_roles(*READ_CASEWORK))) -> Case:
    case = get_case_store().get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"no case with id {case_id!r}")
    return case


# --- A2A agents (triage + investigation) ------------------------------------
# Mounted in this process so the investigation agent's case proposals land in
# the same ApprovalStore the decision endpoint above reads -- agents can
# propose, only an authenticated human reviewer can approve. See
# agents/server.py.


@app.middleware("http")
async def _authenticate_a2a_calls(request: Request, call_next):
    """A2A JSON-RPC endpoints require a bearer token from a role allowed to
    call agents (agent cards stay public for discovery). The principal and
    token are placed in context so proposals are attributed to the caller
    and the triage agent can call the investigation agent on its behalf."""
    if request.method != "POST" or request.url.path not in A2A_RPC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return JSONResponse({"detail": "authentication required"}, 401, {"WWW-Authenticate": "Bearer"})
    try:
        principal = await run_in_threadpool(authenticate_token, token)
    except tokens.InvalidToken:
        return JSONResponse({"detail": "invalid or expired token"}, 401, {"WWW-Authenticate": "Bearer"})
    if principal.role not in CALL_AGENTS:
        await run_in_threadpool(deny, principal, f"POST {request.url.path}")
        return JSONResponse({"detail": "insufficient role for this operation"}, 403)
    current_principal.set(principal)
    current_token.set(token)
    return await call_next(request)


def _record_investigation(result: InvestigationResult) -> None:
    principal = current_principal.get()
    _investigation_results().save(result, requested_by=principal.username if principal else None)


a2a_agents = mount_a2a_agents(
    app, llm_client_factory=get_llm_client, on_investigation_result=_record_investigation
)
