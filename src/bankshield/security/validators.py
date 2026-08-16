"""Structural input validation for the investigation agent's tools.

`tools.py` declares a JSON Schema per tool for the model's benefit
(`TOOL_SPECS`), but nothing previously enforced that schema server-side --
`execute_tool` trusted whatever `input` dict arrived, whether from a real
Bedrock tool-use response or a compromised/jailbroken model. This module
is the server-side enforcement point: an explicit Pydantic model per tool,
with `extra="forbid"` (no smuggled-in extra arguments), bounded string
lengths, and enum-constrained fields (e.g. `create_case`'s `disposition`).

This is a structural control, not a keyword filter: it rejects anything
that doesn't match the declared shape, regardless of what the strings
inside it say.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Shared bounds, reused by the API layer (api/app.py) so HTTP request
# bodies are bounded the same way tool inputs are.
MAX_ID_LEN = 128
MAX_QUERY_LEN = 500
MAX_TEXT_LEN = 4000
MAX_REVIEWER_LEN = 200
MAX_NOTES_LEN = 2000
MAX_CITED_EVIDENCE_ITEMS = 50


class ToolInputValidationError(Exception):
    """Raised when a tool call's arguments don't match that tool's declared
    schema. Callers must treat this as a fail-closed rejection of the call,
    never as license to fall back to unvalidated input."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetTransactionInput(_StrictModel):
    transaction_id: str = Field(min_length=1, max_length=MAX_ID_LEN)


class GetRiskScoreInput(_StrictModel):
    transaction_id: str = Field(min_length=1, max_length=MAX_ID_LEN)


class GetAuthHistoryInput(_StrictModel):
    customer_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    limit: int = Field(default=20, ge=1, le=200)


class GetGraphNeighborsInput(_StrictModel):
    customer_id: str = Field(min_length=1, max_length=MAX_ID_LEN)


class SearchPolicyInput(_StrictModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = Field(default=4, ge=1, le=20)


class CreateCaseInput(_StrictModel):
    transaction_id: str = Field(min_length=1, max_length=MAX_ID_LEN)
    disposition: Literal["confirmed_fraud", "false_positive", "inconclusive_monitor"]
    summary: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    cited_evidence: list[str] = Field(default_factory=list, max_length=MAX_CITED_EVIDENCE_ITEMS)


TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_transaction": GetTransactionInput,
    "get_risk_score": GetRiskScoreInput,
    "get_auth_history": GetAuthHistoryInput,
    "get_graph_neighbors": GetGraphNeighborsInput,
    "search_policy": SearchPolicyInput,
    "create_case": CreateCaseInput,
}


def validate_tool_input(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Validate `tool_input` against `name`'s declared schema and return a
    normalized dict on success.

    Raises `ToolInputValidationError` for any tool name not in the
    allowlist (`TOOL_INPUT_MODELS`) or any input that fails validation --
    unknown fields, wrong types, out-of-range values, or a disposition
    outside the fixed enum. This is deliberately independent of
    `tools.TOOL_NAMES` / `tools._DISPATCH`: a tool only executes if it is
    both dispatchable *and* has a registered, satisfied input schema here.
    """
    if not isinstance(tool_input, dict):
        raise ToolInputValidationError(f"tool input must be an object, got {type(tool_input).__name__}")

    model = TOOL_INPUT_MODELS.get(name)
    if model is None:
        raise ToolInputValidationError(f"no validated input schema registered for tool {name!r}")

    try:
        validated = model.model_validate(tool_input)
    except ValidationError as exc:
        raise ToolInputValidationError(f"invalid arguments for tool {name!r}: {exc}") from exc

    return validated.model_dump()
