"""Append-only audit trail for security-relevant events: logins, approval
requests and decisions, case creation, and authorization denials.

Rows are only ever inserted -- nothing in the codebase updates or deletes
them. Events that change state (a decision, a case) are written in the same
transaction as the change itself, so the trail can't disagree with the data.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from bankshield.persistence.db import Database, as_utc, audit_log, utcnow


def record(
    conn: Connection,
    *,
    action: str,
    outcome: str,
    actor: str | None = None,
    target: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        insert(audit_log).values(
            at=utcnow(), actor=actor, action=action, target=target, outcome=outcome, detail=detail or {}
        )
    )


def record_event(database: Database, **kwargs: Any) -> None:
    with database.transaction() as conn:
        record(conn, **kwargs)


def recent(database: Database, limit: int = 100) -> list[dict[str, Any]]:
    with database.transaction() as conn:
        rows = conn.execute(select(audit_log).order_by(audit_log.c.id.desc()).limit(limit)).mappings().all()
    return [{**row, "at": as_utc(row["at"])} for row in rows]
