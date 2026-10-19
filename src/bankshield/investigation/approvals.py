"""Human-approval gate for consequential actions (POL-CASE-004 §3).

Per policy, an AI investigation assistant may *propose* a consequential
action -- creating a case that names a customer, an account restriction, a
SAR referral -- but must never execute one. This module is the mechanism
that enforces that split in code, not just in a prompt: `request` records a
proposal and returns immediately without side effects; only
`decide(..., approved=True)`, reached from the reviewer-only API endpoint,
causes the downstream action (case creation) to actually happen.

Storage is durable (persistence/db.py): approvals survive restarts and can
be decided by any API process sharing the database. A decision is one
transaction -- a conditional `pending -> decided` update plus, for an
approved `create_case`, the case insert and the audit rows -- so a proposal
is decided at most once and its case is created exactly when it is
approved, even with concurrent reviewers.

Segregation of duties: the principal who requested a proposal (recorded
from the authenticated caller, see auth/context.py) can never be the one
who approves it.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy import func, insert, select, update

from bankshield.persistence import audit
from bankshield.persistence.db import Database, approvals, as_utc, utcnow
from bankshield.investigation.schemas import ApprovalStatus, PendingApproval

if TYPE_CHECKING:
    from bankshield.investigation.case_store import CaseStore


class SelfApprovalError(PermissionError):
    """The reviewer is the principal who requested this proposal."""


def _row_to_approval(row: Any) -> PendingApproval:
    return PendingApproval(
        approval_id=row["approval_id"],
        transaction_id=row["transaction_id"],
        action=row["action"],
        proposed_input=row["proposed_input"],
        rationale=row["rationale"],
        status=ApprovalStatus(row["status"]),
        requested_by=row["requested_by"],
        created_at=as_utc(row["created_at"]),
        decided_at=as_utc(row["decided_at"]),
        reviewer=row["reviewer"],
        decision_notes=row["decision_notes"],
    )


class ApprovalStore:
    """Durable approval queue.

    `ApprovalStore()` with no arguments gets a private in-memory database
    (isolated -- what tests and red-team cases use); the process-wide store
    from `get_approval_store()` shares the configured database with the
    case store.
    """

    def __init__(self, database: Database | None = None, case_store: "CaseStore | None" = None) -> None:
        from bankshield.investigation.case_store import CaseStore

        self.database = database or Database()
        self.case_store = case_store or CaseStore(self.database)
        # Optional in-process hooks (kept for callers that wire their own
        # downstream action); without one, an approved `create_case` creates
        # its case from the stored proposal, which also works after a restart.
        self._on_approve: dict[str, Callable[[PendingApproval], Any]] = {}

    def request(
        self,
        transaction_id: str,
        action: str,
        proposed_input: dict[str, Any],
        rationale: str,
        on_approve: Callable[[PendingApproval], Any] | None = None,
        requested_by: str | None = None,
    ) -> PendingApproval:
        approval_id = f"apr_{uuid.uuid4().hex[:12]}"
        with self.database.transaction() as conn:
            conn.execute(
                insert(approvals).values(
                    approval_id=approval_id,
                    transaction_id=transaction_id,
                    action=action,
                    proposed_input=proposed_input,
                    rationale=rationale,
                    status=ApprovalStatus.PENDING.value,
                    requested_by=requested_by,
                    created_at=utcnow(),
                )
            )
            audit.record(
                conn, action="approval.requested", outcome="ok", actor=requested_by, target=approval_id,
                detail={"transaction_id": transaction_id, "action": action},
            )
        if on_approve is not None:
            self._on_approve[approval_id] = on_approve
        return self.get(approval_id)

    def get(self, approval_id: str) -> PendingApproval | None:
        with self.database.transaction() as conn:
            row = conn.execute(select(approvals).where(approvals.c.approval_id == approval_id)).mappings().first()
        return _row_to_approval(row) if row else None

    def list_pending(self) -> list[PendingApproval]:
        with self.database.transaction() as conn:
            rows = conn.execute(
                select(approvals)
                .where(approvals.c.status == ApprovalStatus.PENDING.value)
                .order_by(approvals.c.created_at)
            ).mappings().all()
        return [_row_to_approval(r) for r in rows]

    def count(self) -> int:
        with self.database.transaction() as conn:
            return conn.execute(select(func.count()).select_from(approvals)).scalar_one()

    def decide(
        self, approval_id: str, approved: bool, reviewer: str, notes: str | None = None
    ) -> PendingApproval:
        with self.database.transaction() as conn:
            row = conn.execute(select(approvals).where(approvals.c.approval_id == approval_id)).mappings().first()
        if row is None:
            raise KeyError(f"no pending approval with id {approval_id!r}")
        if row["status"] != ApprovalStatus.PENDING.value:
            raise ValueError(f"approval {approval_id!r} already decided ({row['status']})")
        if row["requested_by"] and row["requested_by"] == reviewer:
            audit.record_event(self.database, action="approval.decided", outcome="denied", actor=reviewer,
                               target=approval_id, detail={"reason": "self-approval"})
            raise SelfApprovalError(f"{reviewer!r} requested approval {approval_id!r} and cannot also decide it")

        hook = self._on_approve.pop(approval_id, None)
        with self.database.transaction() as conn:
            status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            # Conditional update: only one decision can ever win, even if two
            # reviewers (or two API processes) race on the same approval.
            won = conn.execute(
                update(approvals)
                .where(approvals.c.approval_id == approval_id, approvals.c.status == ApprovalStatus.PENDING.value)
                .values(status=status.value, reviewer=reviewer, decision_notes=notes, decided_at=utcnow())
            ).rowcount
            if won != 1:
                raise ValueError(f"approval {approval_id!r} already decided")
            audit.record(conn, action="approval.decided", outcome=status.value, actor=reviewer, target=approval_id,
                         detail={"transaction_id": row["transaction_id"], "notes": notes})

            decided = _row_to_approval(
                conn.execute(select(approvals).where(approvals.c.approval_id == approval_id)).mappings().one()
            )
            if approved and hook is None and decided.action == "create_case":
                # Same transaction as the decision: the case exists iff the
                # approval is approved.
                self.case_store.create_from_approval(decided, conn=conn)

        if approved and hook is not None:
            hook(decided)
        return decided


_default_store: ApprovalStore | None = None


def get_approval_store() -> ApprovalStore:
    """Process-wide store on the configured database, sharing it with
    `case_store.get_case_store()`. A dedicated instance can be constructed
    directly for isolated tests."""
    global _default_store
    if _default_store is None:
        from bankshield.investigation.case_store import get_case_store
        from bankshield.persistence.db import get_database

        _default_store = ApprovalStore(get_database(), get_case_store())
    return _default_store
