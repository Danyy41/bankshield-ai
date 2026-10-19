"""Triage agent: scores a batch of alerts and escalates by tier.

Input: one data part `{"transaction_ids": [...]}` (1-20 IDs).
For each transaction it calls the read-only `get_risk_score` tool (through
`tools.execute_tool`, i.e. the Phase 5 validators) and applies the
POL-AML-001 §2.1 tiers: Tier 1 and Tier 2 alerts require review, so they
are escalated to the investigation agent **over A2A**; Tier 3 alerts are
recorded as monitored (weekly batch review).

The triage agent holds no consequential capability at all: its tool
allowlist is read-only, it never sees the approval or case stores, and it
only relays the investigation agent's report -- including any pending case
proposals, which still need a named human reviewer's decision.
"""

from __future__ import annotations

from typing import Annotated, Callable

import a2a.types as a2a_types
import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, get_message_text, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from bankshield.agents.common import CARD_PATH, parse_request
from bankshield.agents.investigator import HUMAN_STEP, InvestigationSummary
from bankshield.agents.tasks import run_task
from bankshield.investigation import tools
from bankshield.security.validators import MAX_ID_LEN

TRIAGE_TOOLS = frozenset({"get_risk_score"})
ESCALATED_TIERS = ("tier_1", "tier_2")
MAX_BATCH = 20

if TRIAGE_TOOLS & tools.CONSEQUENTIAL_TOOLS:
    raise RuntimeError("the triage agent must only hold read-only tools")


TransactionId = Annotated[str, StringConstraints(min_length=1, max_length=MAX_ID_LEN)]


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_ids: list[TransactionId] = Field(min_length=1, max_length=MAX_BATCH)


class InvestigatorClient:
    """A2A client for the investigation agent. Resolves its agent card, sends
    one `SendMessage` per escalated alert, and validates the reply against
    `InvestigationSummary` before the triage agent uses any of it."""

    def __init__(self, url: str, httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        self.url = url
        self.httpx_client_factory = httpx_client_factory or (lambda: httpx.AsyncClient(timeout=120))

    async def investigate(self, transaction_id: str) -> InvestigationSummary:
        async with self.httpx_client_factory() as http:
            client = await create_client(
                self.url,
                client_config=ClientConfig(streaming=False, httpx_client=http),
                relative_card_path=CARD_PATH,
            )
            message = new_data_message({"transaction_id": transaction_id}, role=a2a_types.Role.ROLE_USER)
            request = a2a_types.SendMessageRequest(message=message)
            task = None
            async for response in client.send_message(request):
                if response.HasField("task"):
                    task = response.task
        if task is None:
            raise RuntimeError("investigation agent returned no task")
        if task.status.state != a2a_types.TaskState.TASK_STATE_COMPLETED:
            state = a2a_types.TaskState.Name(task.status.state)
            reason = get_message_text(task.status.message) if task.status.HasField("message") else ""
            raise InvestigationFailed(f"{state}: {reason}".rstrip(": "))
        (data,) = get_data_parts(task.artifacts[0].parts)
        try:
            return InvestigationSummary.model_validate(data)
        except ValidationError as exc:
            raise InvestigationFailed("investigation agent returned an unexpected result shape") from exc


class InvestigationFailed(Exception):
    pass


class TriageAgentExecutor(AgentExecutor):
    def __init__(self, investigator: InvestigatorClient) -> None:
        self.investigator = investigator

    async def _triage_one(self, transaction_id: str) -> dict:
        scored = tools.execute_tool("get_risk_score", {"transaction_id": transaction_id})
        if scored.record.is_error:
            return {"transaction_id": transaction_id, "error": scored.output["error"]}
        tier = scored.output["risk_tier"]
        item = {
            "transaction_id": transaction_id,
            "risk_score": scored.output["risk_score"],
            "risk_tier": tier,
            "escalated": tier in ESCALATED_TIERS,
        }
        if not item["escalated"]:
            item["action"] = "monitor (Tier 3 weekly batch review, POL-AML-001 §2.1)"
            return item
        try:
            item["investigation"] = (await self.investigator.investigate(transaction_id)).model_dump()
        except InvestigationFailed as exc:
            item["investigation_error"] = str(exc)
        return item

    async def _triage(self, context: RequestContext) -> dict:
        request = parse_request(context.message, TriageRequest)
        items = [await self._triage_one(txn_id) for txn_id in request.transaction_ids]
        pending = [
            approval["approval_id"]
            for item in items
            for approval in item.get("investigation", {}).get("pending_approvals", [])
        ]
        return {
            "policy": "POL-AML-001 §2.1",
            "triaged": len(items),
            "escalated": sum(1 for item in items if item.get("escalated")),
            "items": items,
            "pending_approval_ids": pending,
            "human_action_required": bool(pending),
            "next_step": HUMAN_STEP if pending else "No consequential action was proposed.",
        }

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await run_task(context, event_queue, self._triage, artifact_name="triage_report")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("triage runs are short-lived and not cancellable")
