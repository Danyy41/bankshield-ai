"""Short-lived signed access tokens (JWT, HS256).

The signing key comes from `BANKSHIELD_JWT_SECRET` (at least 32 bytes). If
it is unset, a random per-process key is generated and a warning logged:
fine for local development, but tokens then stop working on restart, so
deployments must set it (e.g. from AWS Secrets Manager).

Tokens are verified strictly: fixed algorithm (no `alg: none`, no algorithm
confusion), required `exp`/`iat`/`sub`, and matching issuer and audience.
They carry the role only as a hint; authorization always re-reads the
user's current role and disabled flag from the database, so revoking or
re-roling an account takes effect immediately rather than at token expiry.
A managed identity provider (e.g. Amazon Cognito, OIDC) can replace this
module by issuing RS256 tokens verified against its JWKS.
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from datetime import timedelta

import jwt

from bankshield.auth.users import User
from bankshield.persistence.db import utcnow

SECRET_ENV_VAR = "BANKSHIELD_JWT_SECRET"
TTL_ENV_VAR = "BANKSHIELD_TOKEN_TTL_MINUTES"
ALGORITHM = "HS256"
ISSUER = "bankshield-ai"
AUDIENCE = "bankshield-api"
MIN_SECRET_BYTES = 32

logger = logging.getLogger("bankshield.auth")
_ephemeral_secret: str | None = None


class InvalidToken(Exception):
    pass


def _secret() -> str:
    global _ephemeral_secret
    secret = os.environ.get(SECRET_ENV_VAR)
    if secret:
        if len(secret.encode()) < MIN_SECRET_BYTES:
            raise RuntimeError(f"{SECRET_ENV_VAR} must be at least {MIN_SECRET_BYTES} bytes")
        return secret
    if _ephemeral_secret is None:
        logger.warning("%s is not set; using a random per-process signing key (tokens won't survive a restart)",
                       SECRET_ENV_VAR)
        _ephemeral_secret = secrets.token_urlsafe(48)
    return _ephemeral_secret


def ttl() -> timedelta:
    return timedelta(minutes=int(os.environ.get(TTL_ENV_VAR, "30")))


def issue(user: User) -> str:
    now = utcnow()
    claims = {
        "sub": user.username,
        "role": user.role,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + ttl(),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, _secret(), algorithm=ALGORITHM)


def verify(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            _secret(),
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc
