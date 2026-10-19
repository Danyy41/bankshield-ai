"""MCP server exposing BankShield's read-only investigation tools.

Built on the official MCP Python SDK's low-level `Server`, so the tool
layer is reused as-is rather than re-declared:

- discovery advertises the *enforced* input schema for each tool -- the
  Pydantic model in `security/validators.py` (bounded lengths, ranges,
  `additionalProperties: false`) -- with the description from
  `tools.TOOL_SPECS`;
- every call goes through `tools.execute_tool`, i.e. through the same
  server-side `validate_tool_input` the agent loop relies on, and any
  unexpected exception is reduced to a generic message by
  `guardrails.sanitize_error_response` (detail is logged, never returned).

What is deliberately *not* exposed:

- `create_case`, and anything else in `tools.CONSEQUENTIAL_TOOLS`. Over MCP
  it couldn't be made safe and useful at the same time: approvals and cases
  live in the HTTP API process's in-memory `ApprovalStore`/`CaseStore`
  (see approvals.py), while an MCP stdio server is a separate process, so
  a proposal filed here would sit in a queue no human reviewer can see or
  decide. Case proposals stay on the HTTP API's investigation path, where
  `POST /approvals/{id}/decision` with a named reviewer is the only way a
  case gets created.
- Any approval-decision tool. An MCP caller is typically an AI agent;
  deciding an approval is the human half of the gate.

Exposure is an explicit allowlist (`MCP_TOOL_NAMES`), checked at import
against `tools.CONSEQUENTIAL_TOOLS`, so adding a tool to `tools.py` never
exposes it over MCP by accident.

Run (stdio transport, for local MCP clients):
    python scripts/run_mcp_server.py
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from bankshield.investigation import data_access, tools
from bankshield.security.guardrails import sanitize_error_response
from bankshield.security.validators import TOOL_INPUT_MODELS

logger = logging.getLogger("bankshield.mcp")

SERVER_NAME = "bankshield-ai"
SERVER_VERSION = "0.1.0"

MCP_TOOL_NAMES: tuple[str, ...] = (
    "get_transaction",
    "get_risk_score",
    "get_auth_history",
    "get_graph_neighbors",
    "search_policy",
)

if set(MCP_TOOL_NAMES) & tools.CONSEQUENTIAL_TOOLS:
    raise RuntimeError("consequential tools must never be exposed over MCP")
if not set(MCP_TOOL_NAMES) <= tools.TOOL_NAMES or not set(MCP_TOOL_NAMES) <= set(TOOL_INPUT_MODELS):
    raise RuntimeError("every MCP tool needs a tools.py spec and a registered input validator")

INSTRUCTIONS = (
    "Read-only access to BankShield AI's synthetic financial-crime data: transactions, "
    "model risk scores with feature attributions, login history, graph links between "
    "customers, and the AML/fraud policy corpus (cite results as 'DOC-ID §section'). "
    "These tools cannot create cases or take any other action. Case creation is a "
    "consequential action (POL-CASE-004 §3) that requires a named human reviewer's "
    "approval through the BankShield API and is not available over MCP."
)

_SPECS_BY_NAME = {spec["name"]: spec for spec in tools.TOOL_SPECS}
_READ_ONLY = types.ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


def tool_definitions() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=_SPECS_BY_NAME[name]["description"],
            input_schema=TOOL_INPUT_MODELS[name].model_json_schema(),
            annotations=_READ_ONLY,
        )
        for name in MCP_TOOL_NAMES
    ]


def _result(payload: dict[str, Any], is_error: bool) -> types.CallToolResult:
    # Round-trip through JSON so structured content is plain JSON types
    # (timestamps, numpy scalars) and matches the text content exactly.
    payload = json.loads(json.dumps(payload, default=str))
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        is_error=is_error,
    )


def call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
    if name not in MCP_TOOL_NAMES:
        logger.warning("blocked MCP call to non-exposed tool %r", name)
        return _result({"error": f"tool {name!r} is not available over MCP"}, is_error=True)
    try:
        result = tools.execute_tool(name, arguments if arguments is not None else {})
    except Exception as exc:  # fail closed: never leak internals to the caller
        return _result({"error": sanitize_error_response(exc)["detail"]}, is_error=True)
    return _result(result.output, is_error=result.record.is_error)


async def _on_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return types.ListToolsResult(tools=tool_definitions())


async def _on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    # Tool handlers are sync (pandas/XGBoost); run them off the event loop.
    return await anyio.to_thread.run_sync(call_tool, params.name, params.arguments)


def create_server() -> Server:
    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )


async def _serve_stdio() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    # stdout carries the MCP protocol; all logging goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    try:
        # Load data + model up front: fail at startup with a clear message
        # rather than on the first tool call with a sanitized one.
        store = data_access.get_store()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("loaded %d transactions; serving %d read-only tools over stdio", len(store.transactions), len(MCP_TOOL_NAMES))
    anyio.run(_serve_stdio)


if __name__ == "__main__":
    main()
