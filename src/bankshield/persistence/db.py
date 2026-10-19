"""Database engine and schema for BankShield's durable state.

SQLAlchemy Core over any SQLAlchemy URL, chosen by `BANKSHIELD_DATABASE_URL`:

- default: a SQLite file under `var/` -- zero setup for local runs, and the
  Docker image keeps it on a volume (`/app/var`);
- `sqlite://` (no path): a private in-memory database -- what the
  no-argument `ApprovalStore()` / `CaseStore()` constructors use, so tests
  and red-team cases stay isolated from each other;
- PostgreSQL (e.g. Amazon RDS / Aurora) for a real deployment:
  `postgresql+psycopg://...` with a driver installed. The schema only uses
  portable types (String, Text, JSON, timezone-aware DateTime).

Tables are created on first use (`create_all` is idempotent). Schema
changes beyond additive ones should move to Alembic migrations.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import StaticPool

from bankshield import config

DATABASE_URL_ENV_VAR = "BANKSHIELD_DATABASE_URL"
DEFAULT_DATABASE_PATH = config.ROOT_DIR / "var" / "bankshield.db"

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("username", String(64), primary_key=True),
    Column("password_hash", String(255), nullable=False),
    Column("role", String(16), nullable=False),
    Column("disabled", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

approvals = Table(
    "approvals",
    metadata,
    Column("approval_id", String(32), primary_key=True),
    Column("transaction_id", String(128), nullable=False, index=True),
    Column("action", String(32), nullable=False),
    Column("proposed_input", JSON, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("status", String(16), nullable=False, index=True),
    Column("requested_by", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("decided_at", DateTime(timezone=True)),
    Column("reviewer", String(200)),
    Column("decision_notes", Text),
)

cases = Table(
    "cases",
    metadata,
    Column("case_id", String(32), primary_key=True),
    # Unique: at most one case per approval, enforced by the database even
    # under concurrent decisions.
    Column("approval_id", String(32), nullable=False, unique=True),
    Column("transaction_id", String(128), nullable=False, index=True),
    Column("customer_id", String(128), nullable=False),
    Column("disposition", String(32), nullable=False),
    Column("summary", Text, nullable=False),
    Column("cited_evidence", JSON, nullable=False),
    Column("approved_by", String(200)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

investigations = Table(
    "investigations",
    metadata,
    Column("transaction_id", String(128), primary_key=True),
    Column("result", JSON, nullable=False),
    Column("requested_by", String(64)),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

audit_log = Table(
    "audit_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("actor", String(64)),
    Column("action", String(64), nullable=False),
    Column("target", String(128)),
    Column("outcome", String(16), nullable=False),
    Column("detail", JSON),
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite drops tzinfo on the way back; every stored timestamp is UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


class Database:
    def __init__(self, url: str = "sqlite://") -> None:
        self.url = url
        is_sqlite = url.startswith("sqlite")
        in_memory = url in ("sqlite://", "sqlite:///:memory:")
        kwargs: dict = {}
        if is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if in_memory:
            # One shared connection, or each checkout would see an empty DB.
            kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(url, **kwargs)
        # A single shared in-memory connection must not be used by two
        # threads at once; file/server databases rely on their own locking.
        self._lock = threading.RLock() if in_memory else None

        if is_sqlite:
            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_connection, _record):
                cursor = dbapi_connection.cursor()
                if not in_memory:
                    cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        metadata.create_all(self.engine)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        if self._lock is None:
            with self.engine.begin() as conn:
                yield conn
            return
        with self._lock, self.engine.begin() as conn:
            yield conn


_default: Database | None = None
_default_lock = threading.Lock()


def default_database_url() -> str:
    url = os.environ.get(DATABASE_URL_ENV_VAR)
    if url:
        return url
    DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DATABASE_PATH}"


def get_database() -> Database:
    """Process-wide database, from BANKSHIELD_DATABASE_URL (created lazily)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Database(default_database_url())
        return _default
