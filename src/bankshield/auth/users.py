"""Durable user accounts. Each user has exactly one role (see __init__.py)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from bankshield.auth.passwords import DUMMY_HASH, hash_password, verify_password
from bankshield.persistence.db import Database, get_database, users, utcnow

ROLES = ("analyst", "reviewer", "admin", "agent")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


@dataclass(frozen=True)
class User:
    username: str
    role: str
    disabled: bool


class UserStore:
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()

    def create(self, username: str, password: str, role: str) -> User:
        if not USERNAME_RE.match(username):
            raise ValueError("username must be 3-64 chars of a-z, 0-9, '.', '_', '-'")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        try:
            with self.database.transaction() as conn:
                conn.execute(
                    insert(users).values(
                        username=username, password_hash=hash_password(password), role=role,
                        disabled=False, created_at=utcnow(),
                    )
                )
        except IntegrityError as exc:
            raise ValueError(f"user {username!r} already exists") from exc
        return User(username, role, False)

    def get(self, username: str) -> User | None:
        with self.database.transaction() as conn:
            row = conn.execute(select(users).where(users.c.username == username)).mappings().first()
        return User(row["username"], row["role"], row["disabled"]) if row else None

    def authenticate(self, username: str, password: str) -> User | None:
        with self.database.transaction() as conn:
            row = conn.execute(select(users).where(users.c.username == username)).mappings().first()
        if row is None:
            verify_password(password, DUMMY_HASH)  # equalize timing
            return None
        if not verify_password(password, row["password_hash"]) or row["disabled"]:
            return None
        return User(row["username"], row["role"], row["disabled"])

    def set_disabled(self, username: str, disabled: bool) -> None:
        with self.database.transaction() as conn:
            if conn.execute(update(users).where(users.c.username == username).values(disabled=disabled)).rowcount != 1:
                raise KeyError(f"no user {username!r}")

    def set_password(self, username: str, password: str) -> None:
        with self.database.transaction() as conn:
            changed = conn.execute(
                update(users).where(users.c.username == username).values(password_hash=hash_password(password))
            ).rowcount
            if changed != 1:
                raise KeyError(f"no user {username!r}")

    def list(self) -> list[User]:
        with self.database.transaction() as conn:
            rows = conn.execute(select(users).order_by(users.c.username)).mappings().all()
        return [User(r["username"], r["role"], r["disabled"]) for r in rows]
