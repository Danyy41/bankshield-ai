"""Durable store for the latest investigation result per transaction, so a
reviewer can still open the full evidence (`GET /investigations/{id}`)
behind a pending approval after the API restarts."""

from __future__ import annotations

from sqlalchemy import delete, insert, select

from bankshield.investigation.schemas import InvestigationResult
from bankshield.persistence.db import Database, get_database, investigations, utcnow


class InvestigationResultStore:
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or get_database()

    def save(self, result: InvestigationResult, requested_by: str | None) -> None:
        with self.database.transaction() as conn:
            conn.execute(delete(investigations).where(investigations.c.transaction_id == result.transaction_id))
            conn.execute(
                insert(investigations).values(
                    transaction_id=result.transaction_id,
                    result=result.model_dump(mode="json"),
                    requested_by=requested_by,
                    updated_at=utcnow(),
                )
            )

    def get(self, transaction_id: str) -> InvestigationResult | None:
        with self.database.transaction() as conn:
            row = conn.execute(
                select(investigations.c.result).where(investigations.c.transaction_id == transaction_id)
            ).first()
        return InvestigationResult.model_validate(row[0]) if row else None
