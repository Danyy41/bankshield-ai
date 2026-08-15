"""Controlled tool registry for the investigation agent.

Every tool the agent can call is declared here as a JSON Schema (the shape
Bedrock's Converse `toolConfig` and Claude's tool-use both expect) paired
with a dispatch function. Five tools are read-only and execute immediately.
`create_case` is different: per POL-CASE-004 §3, creating a case that names
a customer is a *consequential action* and must never be executed directly
by the model. Its dispatch function does not create anything -- it files a
`PendingApproval` (approvals.py) and returns that pending state to the
agent, which must report it to the analyst rather than claim the case
exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bankshield.investigation import data_access
from bankshield.investigation.approvals import get_approval_store
from bankshield.investigation.case_store import get_case_store
from bankshield.investigation.rag import search_policy as _search_policy
from bankshield.investigation.schemas import ToolCallRecord

CONSEQUENTIAL_TOOLS = {"create_case"}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_transaction",
        "description": (
            "Look up the raw facts for one transaction by transaction_id: amount, "
            "merchant category, country, device, IP, timing, and how the amount "
            "compares to the customer's own historical average. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
        },
    },
    {
        "name": "get_risk_score",
        "description": (
            "Get the fraud-detection model's risk score (0-1) and risk tier for a "
            "transaction, plus the top contributing features with their signed "
            "contribution to the score (from the model's native feature "
            "attribution). Use this before making any claim about why a "
            "transaction is or isn't risky. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}},
            "required": ["transaction_id"],
        },
    },
    {
        "name": "get_auth_history",
        "description": (
            "Get a customer's recent login/session history (device, IP, country, "
            "success/failure, whether the device/location was new), most recent "
            "first. Use this to check for the account-takeover pattern described "
            "in POL-ATO-003 -- a burst of failed attempts followed by a "
            "suspicious successful login. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "limit": {"type": "integer", "description": "Max events to return, default 20"},
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_graph_neighbors",
        "description": (
            "Get the other customers linked to this customer via a shared device, "
            "IP address, or beneficiary, and whether each linked customer has a "
            "prior confirmed-fraud transaction. Use this to check for mule-ring "
            "structure per POL-GRAPH-006. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search BankShield's AML/fraud/cyber policy and typology documents for "
            "passages relevant to a query. Returns doc_id + section + text for each "
            "match, suitable for citation (e.g. 'POL-ATO-003 §2.1'). Use this to "
            "ground any policy-based claim in a specific, citable passage rather "
            "than asserting policy from memory. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "description": "Number of passages to return, default 4"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_case",
        "description": (
            "Propose creating an investigation case naming this transaction's "
            "customer, with a disposition and a cited-evidence summary. This is a "
            "CONSEQUENTIAL action per POL-CASE-004 §3: calling this tool does NOT "
            "create a case. It files a proposal that a human reviewer must "
            "explicitly approve before anything is created. The result you get "
            "back will always be a pending-approval state, never a created case -- "
            "report it to the analyst as pending, not as done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "disposition": {
                    "type": "string",
                    "enum": ["confirmed_fraud", "false_positive", "inconclusive_monitor"],
                },
                "summary": {
                    "type": "string",
                    "description": "Plain-language rationale citing the specific evidence considered.",
                },
                "cited_evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Policy citations supporting this disposition, e.g. ['POL-ATO-003 §2.1'].",
                },
            },
            "required": ["transaction_id", "disposition", "summary"],
        },
    },
]

TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


class ToolExecutionError(Exception):
    pass


def _tool_get_transaction(input: dict[str, Any]) -> dict:
    return data_access.get_transaction(input["transaction_id"])


def _tool_get_risk_score(input: dict[str, Any]) -> dict:
    return data_access.get_risk_score(input["transaction_id"])


def _tool_get_auth_history(input: dict[str, Any]) -> dict:
    limit = input.get("limit", 20)
    events = data_access.get_auth_history(input["customer_id"], limit=limit)
    return {"customer_id": input["customer_id"], "events": events}


def _tool_get_graph_neighbors(input: dict[str, Any]) -> dict:
    neighbors = data_access.get_graph_neighbors(input["customer_id"])
    return {"customer_id": input["customer_id"], "neighbors": neighbors}


def _tool_search_policy(input: dict[str, Any]) -> dict:
    top_k = input.get("top_k", 4)
    return {"results": _search_policy(input["query"], top_k=top_k)}


def _tool_create_case(input: dict[str, Any]) -> dict:
    transaction_id = input["transaction_id"]
    txn = data_access.get_transaction(transaction_id)
    proposed_input = {
        "transaction_id": transaction_id,
        "customer_id": txn["customer_id"],
        "disposition": input["disposition"],
        "summary": input["summary"],
        "cited_evidence": input.get("cited_evidence", []),
    }
    approval = get_approval_store().request(
        transaction_id=transaction_id,
        action="create_case",
        proposed_input=proposed_input,
        rationale=input["summary"],
        on_approve=lambda approved: get_case_store().create_from_approval(approved),
    )
    return {
        "status": "pending_approval",
        "approval_id": approval.approval_id,
        "message": (
            "Case proposal recorded and awaiting human approval per POL-CASE-004 §3. "
            "No case has been created. Report this as pending, not as completed."
        ),
    }


_DISPATCH = {
    "get_transaction": _tool_get_transaction,
    "get_risk_score": _tool_get_risk_score,
    "get_auth_history": _tool_get_auth_history,
    "get_graph_neighbors": _tool_get_graph_neighbors,
    "search_policy": _tool_search_policy,
    "create_case": _tool_create_case,
}


@dataclass
class ToolExecutionResult:
    output: dict[str, Any]
    record: ToolCallRecord


def execute_tool(name: str, input: dict[str, Any]) -> ToolExecutionResult:
    if name not in _DISPATCH:
        raise ToolExecutionError(f"unknown tool {name!r}")

    start = time.monotonic()
    try:
        output = _DISPATCH[name](input)
        is_error = False
        summary = f"ok ({len(str(output))} chars)"
    except (data_access.NotFoundError, KeyError, ValueError) as exc:
        output = {"error": str(exc)}
        is_error = True
        summary = f"error: {exc}"
    latency_ms = (time.monotonic() - start) * 1000

    record = ToolCallRecord(
        tool_name=name,
        input=input,
        output=output,
        output_summary=summary,
        is_error=is_error,
        latency_ms=latency_ms,
    )
    return ToolExecutionResult(output=output, record=record)
