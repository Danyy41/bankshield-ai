"""Investigation agent: the existing investigation loop, served over A2A.

Input: one data part `{"transaction_id": "..."}` (nothing else accepted).
Output: a task artifact with an `InvestigationSummary` -- disposition,
narrative, verified citations, the tools used, and any *pending* case
proposals.

The agent reuses `investigation.agent.run_investigation` unchanged, so it
inherits every existing control: tool calls go through
`guardrails.safe_execute_tool` and the Phase 5 validators, citations are
filtered to what was actually retrieved, and the only consequential tool,
`create_case`, merely files a `PendingApproval`. Nothing here can decide an
approval: the summary reports pending approvals and the human step they
need, and a case only ever exists after a named reviewer approves it via
`POST /approvals/{approval_id}/decision`.
"""

from __future__ import annotations

from typing import Callable

import anyio
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from pydantic import BaseModel, ConfigDict, Field

from bankshield.agents.common import parse_request
from bankshield.agents.tasks import run_task
from bankshield.investigation.agent import run_investigation
from bankshield.investigation.llm_client import LLMClient
from bankshield.investigation.schemas import InvestigationResult
from bankshield.security.validators import MAX_ID_LEN

HUMAN_STEP = (
    "No case has been created. A named human reviewer must decide each pending approval via "
    "POST /approvals/{approval_id}/decision (POL-CASE-004 §3); agents cannot."
)


class InvestigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=MAX_ID_LEN)


class PendingApprovalRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str
    action: str
    status: str


class InvestigationSummary(BaseModel):
    """What the investigation agent returns over A2A -- and what the triage
    agent validates its replies against, so each agent treats the other's
    output as data of a known shape, not free-form instructions."""

    model_config = ConfigDict(extra="forbid")
    transaction_id: str
    risk_score: float
    risk_tier: str
    disposition: str
    narrative: str
    citations: list[str]
    tools_used: list[str]
    model_id: str
    pending_approvals: list[PendingApprovalRef]
    human_action_required: bool
    next_step: str

    @classmethod
    def from_result(cls, result: InvestigationResult) -> "InvestigationSummary":
        pending = [
            PendingApprovalRef(approval_id=a.approval_id, action=a.action, status=a.status.value)
            for a in result.pending_approvals
        ]
        return cls(
            transaction_id=result.transaction_id,
            risk_score=result.payload.risk.risk_score,
            risk_tier=result.payload.risk.risk_tier.value,
            disposition=result.disposition,
            narrative=result.narrative,
            citations=[f"{c.doc_id} {c.section}" for c in result.citations],
            tools_used=[call.tool_name for call in result.tool_calls],
            model_id=result.model_id,
            pending_approvals=pending,
            human_action_required=bool(pending),
            next_step=HUMAN_STEP if pending else "No consequential action was proposed.",
        )


class InvestigationAgentExecutor(AgentExecutor):
    def __init__(
        self,
        llm_client_factory: Callable[[], LLMClient],
        on_result: Callable[[InvestigationResult], None] | None = None,
    ) -> None:
        self._llm_client_factory = llm_client_factory
        self._on_result = on_result

    async def _investigate(self, context: RequestContext) -> dict:
        request = parse_request(context.message, InvestigateRequest)
        # The investigation loop is synchronous (pandas/XGBoost/LLM client);
        # keep it off the event loop.
        result = await anyio.to_thread.run_sync(
            run_investigation, request.transaction_id, self._llm_client_factory()
        )
        if self._on_result is not None:
            # Lets a reviewer open the full result (payload, tool calls,
            # narrative) at GET /investigations/{transaction_id} before deciding.
            self._on_result(result)
        return InvestigationSummary.from_result(result).model_dump()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await run_task(context, event_queue, self._investigate, artifact_name="investigation")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("investigations are short-lived and not cancellable")
