"""Human-approval gate for consequential actions (POL-CASE-004 §3).

Per policy, an AI investigation assistant may *propose* a consequential
action -- creating a case that names a customer, an account restriction, a
SAR referral -- but must never execute one. This module is the mechanism
that enforces that split in code, not just in a prompt: `request_approval`
records a proposal and returns immediately without side effects; only
`decide_approval(..., approved=True)`, called from a human-facing API
endpoint, causes the downstream action (case creation) to actually happen.

Storage is in-memory, process-local -- adequate for this portfolio service;
a production deployment would back this with a durable store (e.g.
DynamoDB) so the approval queue survives a restart and is auditable across
instances.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from bankshield.investigation.schemas import ApprovalStatus, PendingApproval


@dataclass
class ApprovalStore:
    _approvals: dict[str, PendingApproval] = field(default_factory=dict)
    _on_approve: dict[str, Callable[[PendingApproval], Any]] = field(default_factory=dict)

    def request(
        self,
        transaction_id: str,
        action: str,
        proposed_input: dict[str, Any],
        rationale: str,
        on_approve: Callable[[PendingApproval], Any] | None = None,
    ) -> PendingApproval:
        approval = PendingApproval(
            approval_id=f"apr_{uuid.uuid4().hex[:12]}",
            transaction_id=transaction_id,
            action=action,
            proposed_input=proposed_input,
            rationale=rationale,
            status=ApprovalStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        self._approvals[approval.approval_id] = approval
        if on_approve is not None:
            self._on_approve[approval.approval_id] = on_approve
        return approval

    def get(self, approval_id: str) -> PendingApproval | None:
        return self._approvals.get(approval_id)

    def list_pending(self) -> list[PendingApproval]:
        return [a for a in self._approvals.values() if a.status == ApprovalStatus.PENDING]

    def decide(
        self, approval_id: str, approved: bool, reviewer: str, notes: str | None = None
    ) -> PendingApproval:
        approval = self._approvals.get(approval_id)
        if approval is None:
            raise KeyError(f"no pending approval with id {approval_id!r}")
        if approval.status != ApprovalStatus.PENDING:
            raise ValueError(f"approval {approval_id!r} already decided ({approval.status})")

        approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        approval.reviewer = reviewer
        approval.decision_notes = notes
        approval.decided_at = datetime.now(timezone.utc)

        if approved:
            callback = self._on_approve.pop(approval_id, None)
            if callback is not None:
                callback(approval)
        else:
            self._on_approve.pop(approval_id, None)

        return approval


_default_store = ApprovalStore()


def get_approval_store() -> ApprovalStore:
    """Process-wide singleton, mirroring data_access.get_store()'s pattern.
    A dedicated instance can be constructed directly for isolated tests."""
    return _default_store
