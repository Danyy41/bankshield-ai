"""Durable approvals/cases: restart recovery, atomic decisions, and the
audit trail -- at the store level and against a real API process that is
stopped and restarted."""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from bankshield.auth.users import UserStore
from bankshield.investigation import data_access
from bankshield.investigation.approvals import ApprovalStore, SelfApprovalError
from bankshield.investigation.case_store import CaseStore
from bankshield.investigation.schemas import ApprovalStatus
from bankshield.persistence import audit
from bankshield.persistence.db import Database
from tests.conftest import TEST_PASSWORD

ROOT = Path(__file__).resolve().parents[1]


def _stores(url):
    """A fresh process's view of the database: new engine, new stores."""
    db = Database(url)
    cases = CaseStore(db)
    return db, ApprovalStore(db, cases), cases


def _propose(approvals, requested_by="alice.analyst"):
    return approvals.request(
        transaction_id="txn_1",
        action="create_case",
        proposed_input={"transaction_id": "txn_1", "customer_id": "CUST000001",
                        "disposition": "confirmed_fraud", "summary": "s", "cited_evidence": ["POL-ATO-003 §2.1"]},
        rationale="s",
        requested_by=requested_by,
    )


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path}/state.db"


def test_pending_approval_survives_restart_and_creates_its_case_when_approved(db_url):
    _, approvals, _ = _stores(db_url)
    approval_id = _propose(approvals).approval_id

    # "Restart": nothing in memory carries over -- no callbacks, no objects.
    _, approvals, cases = _stores(db_url)
    assert [a.approval_id for a in approvals.list_pending()] == [approval_id]
    assert cases.list_all() == []
    approvals.decide(approval_id, approved=True, reviewer="rita.reviewer")

    db, approvals, cases = _stores(db_url)
    assert approvals.get(approval_id).status == ApprovalStatus.APPROVED
    (case,) = cases.list_all()
    assert case.approval_id == approval_id and case.approved_by == "rita.reviewer"
    actions = [e["action"] for e in audit.recent(db)]
    assert {"approval.requested", "approval.decided", "case.created"} <= set(actions)


def test_rejected_approval_never_creates_a_case(db_url):
    _, approvals, cases = _stores(db_url)
    approval_id = _propose(approvals).approval_id
    approvals.decide(approval_id, approved=False, reviewer="rita.reviewer")
    assert cases.list_all() == []
    with pytest.raises(ValueError):
        approvals.decide(approval_id, approved=True, reviewer="rob.reviewer")
    assert cases.list_all() == []


def test_concurrent_decisions_from_separate_processes_views_produce_one_case(db_url):
    """Two API processes (two engines on one database) race to decide the
    same approval: exactly one wins and exactly one case exists."""
    _, approvals, _ = _stores(db_url)
    approval_id = _propose(approvals).approval_id
    views = [_stores(db_url) for _ in range(8)]
    outcomes: list[str] = []
    barrier = threading.Barrier(len(views))

    def decide(view, i):
        barrier.wait()
        try:
            view[1].decide(approval_id, approved=True, reviewer=f"reviewer.{i}")
            outcomes.append("won")
        except ValueError:
            outcomes.append("lost")

    threads = [threading.Thread(target=decide, args=(v, i)) for i, v in enumerate(views)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["lost"] * 7 + ["won"]
    assert len(_stores(db_url)[2].list_all()) == 1


def test_self_approval_is_refused_and_audited(db_url):
    db, approvals, cases = _stores(db_url)
    approval_id = _propose(approvals, requested_by="rita.reviewer").approval_id
    with pytest.raises(SelfApprovalError):
        approvals.decide(approval_id, approved=True, reviewer="rita.reviewer")
    assert approvals.get(approval_id).status == ApprovalStatus.PENDING and cases.list_all() == []
    assert any(e["action"] == "approval.decided" and e["outcome"] == "denied" for e in audit.recent(db))


def test_shared_case_store_refuses_an_approval_not_recorded_as_approved(db_url):
    """A caller that builds its own 'approved' approval object can't create a
    case on the shared database: the recorded status is what counts."""
    _, approvals, cases = _stores(db_url)
    pending = _propose(approvals)
    forged = pending.model_copy(update={"status": ApprovalStatus.APPROVED, "reviewer": "nobody"})
    with pytest.raises(ValueError):
        cases.create_from_approval(forged)
    assert cases.list_all() == []


# --- a real API process, stopped and restarted ----------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, db_url):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "BANKSHIELD_DATABASE_URL": db_url,
                    "BANKSHIELD_A2A_BASE_URL": self.base}
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "bankshield.api.app:app", "--host", "127.0.0.1", "--port", str(self.port)],
            env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(120):
            try:
                if httpx.get(self.base + "/health", timeout=1).status_code == 200:
                    return
            except httpx.TransportError:
                time.sleep(0.5)
        pytest.fail("server did not start")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=15)


def test_api_restart_recovery_end_to_end(tmp_path):
    db_url = f"sqlite:///{tmp_path}/api.db"
    store = UserStore(Database(db_url))
    for name, role in [("ana.analyst", "analyst"), ("rev.reviewer", "reviewer"), ("adm.admin", "admin")]:
        store.create(name, TEST_PASSWORD, role)
    txn_id = next(s["transaction_id"] for s in data_access.sample_transactions(n=5) if s["risk_tier"] == "tier_1")

    def login(base, username):
        token = httpx.post(base + "/auth/token", data={"username": username, "password": TEST_PASSWORD}).json()
        return {"Authorization": f"Bearer {token['access_token']}"}

    server = _Server(db_url)
    server.start()
    try:
        analyst, reviewer = login(server.base, "ana.analyst"), login(server.base, "rev.reviewer")
        investigation = httpx.post(server.base + "/investigations", json={"transaction_id": txn_id},
                                   headers=analyst, timeout=60).json()
        (approval,) = investigation["pending_approvals"]
    finally:
        server.stop()

    # Restart 1: the pending approval, the investigation, and (same signing
    # key) the reviewer's existing token all survive.
    server.start()
    try:
        pending = httpx.get(server.base + "/approvals", headers=reviewer).json()
        assert [a["approval_id"] for a in pending] == [approval["approval_id"]]
        assert pending[0]["requested_by"] == "ana.analyst"
        assert httpx.get(f"{server.base}/investigations/{txn_id}", headers=reviewer).status_code == 200
        assert httpx.get(server.base + "/cases", headers=reviewer).json() == []
        decided = httpx.post(f"{server.base}/approvals/{approval['approval_id']}/decision",
                             json={"approved": True, "notes": "verified"}, headers=reviewer)
        assert decided.status_code == 200
    finally:
        server.stop()

    # Restart 2: the decision and its case are durable, and so is the trail.
    server.start()
    try:
        (case,) = httpx.get(server.base + "/cases", headers=reviewer).json()
        assert case["approval_id"] == approval["approval_id"] and case["approved_by"] == "rev.reviewer"
        assert httpx.get(server.base + "/approvals", headers=reviewer).json() == []
        trail = httpx.get(server.base + "/audit", headers=login(server.base, "adm.admin")).json()
        actions = {e["action"] for e in trail}
        assert {"auth.login", "approval.requested", "approval.decided", "case.created"} <= actions
    finally:
        server.stop()
