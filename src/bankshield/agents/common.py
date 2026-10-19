"""Shared A2A plumbing for the BankShield agents: agent cards, strict
inbound-message parsing, and the public base URL.

Inbound A2A messages are untrusted input, exactly like an HTTP request
body or a model's tool call. Each agent accepts one shape only -- a single
JSON data part matching a strict Pydantic model (`extra="forbid"`, bounded
lengths) -- and rejects everything else, including free-text parts. No
instruction carried in a message can therefore reach agent logic as
anything but a validated identifier.
"""

from __future__ import annotations

import os
from typing import TypeVar

import a2a.types as a2a_types
from a2a.client.client_factory import TransportProtocol
from a2a.helpers import get_data_parts
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT
from pydantic import BaseModel, ValidationError

from bankshield.security.validators import ToolInputValidationError

BASE_URL_ENV_VAR = "BANKSHIELD_A2A_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
CARD_PATH = "/.well-known/agent-card.json"
DATA_MEDIA_TYPE = "application/json"

ModelT = TypeVar("ModelT", bound=BaseModel)


def base_url() -> str:
    """Where this process's agents are reachable -- advertised in the agent
    cards, and used by the triage agent to reach the investigation agent."""
    return os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL).rstrip("/")


def build_agent_card(
    *, name: str, description: str, url: str, skill: a2a_types.AgentSkill
) -> a2a_types.AgentCard:
    return a2a_types.AgentCard(
        name=name,
        description=description,
        version="0.1.0",
        provider=a2a_types.AgentProvider(organization="BankShield AI", url="https://github.com/Danyy41/bankshield-ai"),
        supported_interfaces=[
            a2a_types.AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=a2a_types.AgentCapabilities(streaming=False, push_notifications=False),
        security_schemes={
            "bearer": a2a_types.SecurityScheme(
                http_auth_security_scheme=a2a_types.HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="JWT",
                    description="BankShield access token from POST /auth/token (roles: analyst, agent, admin)",
                )
            )
        },
        security_requirements=[a2a_types.SecurityRequirement(schemes={"bearer": a2a_types.StringList(list=[])})],
        default_input_modes=[DATA_MEDIA_TYPE],
        default_output_modes=[DATA_MEDIA_TYPE],
        skills=[skill],
    )


def parse_request(message: a2a_types.Message | None, model: type[ModelT]) -> ModelT:
    """Accept exactly one data part, validated against `model`; reject
    anything else (text parts, files, extra parts, extra fields)."""
    if message is None or message.role != a2a_types.Role.ROLE_USER:
        raise ToolInputValidationError("expected a user-role message")
    if len(message.parts) != 1 or not message.parts[0].HasField("data"):
        raise ToolInputValidationError("expected exactly one JSON data part")
    (data,) = get_data_parts(message.parts)
    if not isinstance(data, dict):
        raise ToolInputValidationError("data part must be a JSON object")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ToolInputValidationError(f"invalid request: {exc}") from exc
