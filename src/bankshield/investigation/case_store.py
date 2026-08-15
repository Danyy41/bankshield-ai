"""In-memory case store.

A `Case` is only ever created by `ApprovalStore.decide(..., approved=True)`
calling back into `create_case_from_approval` -- there is no code path that
creates a Case directly from a tool call. See approvals.py and
POL-CASE-004 §3.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bankshield.investigation.schemas import Case, PendingApproval


@dataclass
class CaseStore:
    _cases: dict[str, Case] = field(default_factory=dict)

    def create_from_approval(self, approval: PendingApproval) -> Case:
        proposed = approval.proposed_input
        case = Case(
            case_id=f"case_{uuid.uuid4().hex[:12]}",
            transaction_id=approval.transaction_id,
            customer_id=proposed.get("customer_id", ""),
            disposition=proposed.get("disposition", "inconclusive_monitor"),
            summary=proposed.get("summary", ""),
            cited_evidence=proposed.get("cited_evidence", []),
            approval_id=approval.approval_id,
            created_at=datetime.now(timezone.utc),
        )
        self._cases[case.case_id] = case
        return case

    def get(self, case_id: str) -> Case | None:
        return self._cases.get(case_id)

    def list_all(self) -> list[Case]:
        return list(self._cases.values())


_default_store = CaseStore()


def get_case_store() -> CaseStore:
    return _default_store
