"""FastAPI guards: authenticate the bearer token, then authorize by role.

Every protected endpoint declares the roles it accepts via `require_roles`.
A request is rejected with 401 if the token is missing, malformed, expired,
forged, or belongs to an unknown or disabled user, and with 403 (audited)
if the user's *current* role -- re-read from the database on every request
-- isn't allowed. On success the principal is placed in
`auth.context.current_principal` for code deeper in the request.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from bankshield.auth import tokens
from bankshield.auth.context import Principal, current_principal, current_token
from bankshield.auth.users import UserStore
from bankshield.persistence import audit
from bankshield.persistence.db import get_database

ANALYST, REVIEWER, ADMIN, AGENT = "analyst", "reviewer", "admin", "agent"

# Who may do what. Deciding approvals is deliberately reviewer-only: admins
# and agents can't, and nothing else in the API can create a case.
READ_DATA = (ANALYST, REVIEWER, ADMIN, AGENT)
RUN_INVESTIGATION = (ANALYST, REVIEWER, ADMIN)
READ_CASEWORK = (ANALYST, REVIEWER, ADMIN)
DECIDE_APPROVAL = (REVIEWER,)
READ_AUDIT = (ADMIN,)
CALL_AGENTS = (ANALYST, ADMIN, AGENT)

_bearer = HTTPBearer(auto_error=False)
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def authenticate_token(token: str) -> Principal:
    """Verify a bearer token and resolve it to a current, enabled user."""
    claims = tokens.verify(token)
    user = UserStore().get(claims["sub"])
    if user is None or user.disabled:
        raise tokens.InvalidToken("unknown or disabled user")
    return Principal(user.username, user.role)


def deny(principal: Principal, target: str) -> None:
    audit.record_event(get_database(), action="access.denied", outcome="denied", actor=principal.username,
                       target=target, detail={"role": principal.role})


def require_roles(*roles: str):
    async def guard(
        request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)
    ) -> Principal:
        if credentials is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required", _UNAUTHORIZED_HEADERS)
        try:
            principal = await run_in_threadpool(authenticate_token, credentials.credentials)
        except tokens.InvalidToken:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token", _UNAUTHORIZED_HEADERS)
        if principal.role not in roles:
            await run_in_threadpool(deny, principal, f"{request.method} {request.url.path}")
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role for this operation")
        # Async guard on purpose: context variables set here are inherited by
        # the endpoint (and the threadpool it runs in).
        current_principal.set(principal)
        current_token.set(credentials.credentials)
        return principal

    return guard


class LoginThrottle:
    """Lock a username out for `lockout_seconds` after `max_failures`
    consecutive failed logins -- blunts online password guessing. Per
    process; a multi-instance deployment would back this with the database
    or rely on the identity provider's own protections."""

    def __init__(self, max_failures: int = 5, lockout_seconds: int = 300) -> None:
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self._failures: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def locked(self, username: str) -> bool:
        with self._lock:
            recent = [t for t in self._failures[username] if time.monotonic() - t < self.lockout_seconds]
            self._failures[username] = recent
            return len(recent) >= self.max_failures

    def failed(self, username: str) -> None:
        with self._lock:
            self._failures[username].append(time.monotonic())

    def succeeded(self, username: str) -> None:
        with self._lock:
            self._failures.pop(username, None)


login_throttle = LoginThrottle()
