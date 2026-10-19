"""The authenticated principal for the current request.

Set by the API's auth guards and the A2A middleware; read by code deeper in
the call stack that must attribute an action -- e.g. `tools.create_case`
records who requested a case proposal, which is what lets the approval gate
refuse self-approval. Context variables follow the request into threadpool
workers and the A2A executor's tasks.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    username: str
    role: str


current_principal: ContextVar[Principal | None] = ContextVar("bankshield_principal", default=None)
# The caller's bearer token, so the triage agent can call the investigation
# agent on the caller's behalf (delegation) instead of with its own authority.
current_token: ContextVar[str | None] = ContextVar("bankshield_token", default=None)


def requesting_username() -> str | None:
    principal = current_principal.get()
    return principal.username if principal else None
