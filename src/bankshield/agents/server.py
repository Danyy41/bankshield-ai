"""Mount the triage and investigation agents on the BankShield FastAPI app.

Each agent gets an A2A JSON-RPC endpoint and an agent card:

    POST /a2a/triage                                  JSON-RPC (SendMessage, GetTask, ...)
    GET  /a2a/triage/.well-known/agent-card.json
    POST /a2a/investigator
    GET  /a2a/investigator/.well-known/agent-card.json

Mounting them in the API process -- rather than as separate services -- is
what keeps the human-approval gate usable: the investigation agent's case
proposals land in the same in-memory `ApprovalStore` that
`POST /approvals/{approval_id}/decision` reads. The triage agent still
reaches the investigation agent over HTTP with the A2A client, so the two
agents could be split into separate services once the approval/case store
is durable and shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import a2a.types as a2a_types
import httpx
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI

from bankshield.agents.common import CARD_PATH, base_url, build_agent_card
from bankshield.agents.investigator import InvestigationAgentExecutor
from bankshield.agents.triage import MAX_BATCH, InvestigatorClient, TriageAgentExecutor
from bankshield.investigation.llm_client import LLMClient
from bankshield.investigation.schemas import InvestigationResult

TRIAGE_PATH = "/a2a/triage"
INVESTIGATOR_PATH = "/a2a/investigator"


@dataclass
class MountedAgents:
    triage_card: a2a_types.AgentCard
    investigator_card: a2a_types.AgentCard
    investigator_client: InvestigatorClient


def mount_a2a_agents(
    app: FastAPI,
    *,
    llm_client_factory: Callable[[], LLMClient],
    on_investigation_result: Callable[[InvestigationResult], None] | None = None,
    httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> MountedAgents:
    base = base_url()
    investigator_card = build_agent_card(
        name="BankShield Investigation Agent",
        description=(
            "Investigates one alert end to end with BankShield's read-only tools and policy "
            "corpus, returning a cited disposition. May propose a case, which only ever becomes "
            "a pending approval for a named human reviewer -- it cannot create or approve cases."
        ),
        url=base + INVESTIGATOR_PATH,
        skill=a2a_types.AgentSkill(
            id="investigate_transaction",
            name="Investigate a transaction",
            description='Input: one JSON data part {"transaction_id": "..."}.',
            tags=["aml", "fraud", "investigation"],
            examples=['{"transaction_id": "849e00a5ee8c483ba5751e12007432b0"}'],
        ),
    )
    triage_card = build_agent_card(
        name="BankShield Triage Agent",
        description=(
            "Scores a batch of alerts with the fraud model, applies the POL-AML-001 §2.1 tiers, "
            "and escalates Tier 1/2 alerts to the Investigation Agent over A2A. Read-only."
        ),
        url=base + TRIAGE_PATH,
        skill=a2a_types.AgentSkill(
            id="triage_alerts",
            name="Triage alerts",
            description=f'Input: one JSON data part {{"transaction_ids": [...]}} with 1-{MAX_BATCH} IDs.',
            tags=["aml", "fraud", "triage"],
            examples=['{"transaction_ids": ["849e00a5ee8c483ba5751e12007432b0"]}'],
        ),
    )
    investigator_client = InvestigatorClient(base + INVESTIGATOR_PATH, httpx_client_factory)

    for path, card, executor in (
        (INVESTIGATOR_PATH, investigator_card, InvestigationAgentExecutor(llm_client_factory, on_investigation_result)),
        (TRIAGE_PATH, triage_card, TriageAgentExecutor(investigator_client)),
    ):
        handler = DefaultRequestHandlerV2(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
        app.router.routes.extend(create_jsonrpc_routes(handler, rpc_url=path))
        app.router.routes.extend(create_agent_card_routes(card, card_url=path + CARD_PATH))

    return MountedAgents(triage_card, investigator_card, investigator_client)
