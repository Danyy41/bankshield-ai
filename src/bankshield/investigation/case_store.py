"""Durable case store.

A `Case` is only ever created from an *approved* approval -- normally by
`ApprovalStore.decide(..., approved=True)`, in the same transaction as the
decision. There is no code path that creates a Case directly from a tool
call. See approvals.py and POL-CASE-004 §3.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from bankshield.persistence import audit
from bankshield.persistence.db import Database, approvals, as_utc, cases, utcnow
from bankshield.investigation.schemas import ApprovalStatus, Case, PendingApproval


def _row_to_case(row: Any) -> Case:
    return Case(
        case_id=row["case_id"],
        transaction_id=row["transaction_id"],
        customer_id=row["customer_id"],
        disposition=row["disposition"],
        summary=row["summary"],
        cited_evidence=row["cited_evidence"],
        approval_id=row["approval_id"],
        approved_by=row["approved_by"],
        created_at=as_utc(row["created_at"]),
    )


class CaseStore:
    """`CaseStore()` with no arguments gets a private in-memory database;
    the process-wide store from `get_case_store()` uses the configured one.

    On a shared database (`require_recorded_approval`), the approval must
    also exist there with status approved -- a caller can't create a case
    from an approval object it constructed itself.
    """

    def __init__(self, database: Database | None = None) -> None:
        self.require_recorded_approval = database is not None
        self.database = database or Database()

    def create_from_approval(self, approval: PendingApproval, conn: Connection | None = None) -> Case:
        # Defense in depth: the only production caller is
        # ApprovalStore.decide(approved=True), which already guarantees this
        # -- but a case must never be creatable from an approval that isn't
        # actually approved, so the invariant is asserted here too.
        if approval.status != ApprovalStatus.APPROVED:
            raise ValueError(
                f"refusing to create a case from a non-approved approval (status={approval.status!r})"
            )
        if conn is None:
            with self.database.transaction() as own_conn:
                return self._insert(approval, own_conn)
        return self._insert(approval, conn)

    def _insert(self, approval: PendingApproval, conn: Connection) -> Case:
        if self.require_recorded_approval:
            recorded = conn.execute(
                select(approvals.c.status).where(approvals.c.approval_id == approval.approval_id)
            ).scalar_one_or_none()
            if recorded != ApprovalStatus.APPROVED.value:
                raise ValueError(
                    f"refusing to create a case: approval {approval.approval_id!r} is not recorded as approved"
                )
        proposed = approval.proposed_input
        case_id = f"case_{uuid.uuid4().hex[:12]}"
        conn.execute(
            insert(cases).values(
                case_id=case_id,
                approval_id=approval.approval_id,
                transaction_id=approval.transaction_id,
                customer_id=proposed.get("customer_id", ""),
                disposition=proposed.get("disposition", "inconclusive_monitor"),
                summary=proposed.get("summary", ""),
                cited_evidence=proposed.get("cited_evidence", []),
                approved_by=approval.reviewer,
                created_at=utcnow(),
            )
        )
        audit.record(conn, action="case.created", outcome="ok", actor=approval.reviewer, target=case_id,
                     detail={"approval_id": approval.approval_id, "transaction_id": approval.transaction_id})
        return _row_to_case(conn.execute(select(cases).where(cases.c.case_id == case_id)).mappings().one())

    def get(self, case_id: str) -> Case | None:
        with self.database.transaction() as conn:
            row = conn.execute(select(cases).where(cases.c.case_id == case_id)).mappings().first()
        return _row_to_case(row) if row else None

    def list_all(self) -> list[Case]:
        with self.database.transaction() as conn:
            rows = conn.execute(select(cases).order_by(cases.c.created_at)).mappings().all()
        return [_row_to_case(r) for r in rows]


_default_store: CaseStore | None = None


def get_case_store() -> CaseStore:
    global _default_store
    if _default_store is None:
        from bankshield.persistence.db import get_database

        _default_store = CaseStore(get_database())
    return _default_store
