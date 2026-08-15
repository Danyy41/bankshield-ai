"""End-to-end tests for the Phase 4 FastAPI service layer, driven by
AutoFakeLLMClient so no AWS credentials are required."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bankshield.investigation import data_access
from bankshield.investigation.llm_client import AutoFakeLLMClient


@pytest.fixture(scope="module")
def client():
    from bankshield.api.app import app, get_llm_client

    app.dependency_overrides[get_llm_client] = lambda: AutoFakeLLMClient()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def tier1_transaction_id():
    store = data_access.get_store()
    candidates = store.transactions[
        (store.transactions["is_fraud"] == 1) & (store.transactions["graph_suspicious_neighbor_count"] > 0)
    ]
    for _, row in candidates.iterrows():
        risk = data_access.get_risk_score(row["transaction_id"])
        if risk["risk_tier"] == "tier_1":
            return row["transaction_id"]
    pytest.skip("no tier_1 transaction available for API tests")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_transaction(client, tier1_transaction_id):
    response = client.get(f"/transactions/{tier1_transaction_id}")
    assert response.status_code == 200
    assert response.json()["transaction_id"] == tier1_transaction_id


def test_get_transaction_404(client):
    response = client.get("/transactions/no-such-id")
    assert response.status_code == 404


def test_get_risk_score(client, tier1_transaction_id):
    response = client.get(f"/transactions/{tier1_transaction_id}/risk")
    assert response.status_code == 200
    body = response.json()
    assert body["risk_tier"] == "tier_1"


def test_get_auth_history(client, tier1_transaction_id):
    txn = client.get(f"/transactions/{tier1_transaction_id}").json()
    response = client.get(f"/customers/{txn['customer_id']}/auth-history")
    assert response.status_code == 200
    assert "events" in response.json()


def test_get_graph_neighbors(client, tier1_transaction_id):
    txn = client.get(f"/transactions/{tier1_transaction_id}").json()
    response = client.get(f"/customers/{txn['customer_id']}/graph-neighbors")
    assert response.status_code == 200
    assert "neighbors" in response.json()


def test_policy_search(client):
    response = client.post("/policy/search", json={"query": "account takeover", "top_k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) <= 2
    assert all(r["doc_id"].startswith("POL-") for r in results)


def test_full_investigation_to_case_workflow(client, tier1_transaction_id):
    # 1. Run the investigation -- proposes a case, creates nothing yet.
    response = client.post("/investigations", json={"transaction_id": tier1_transaction_id})
    assert response.status_code == 200
    body = response.json()
    assert body["disposition"] == "confirmed_fraud"
    assert len(body["pending_approvals"]) == 1
    assert body["pending_approvals"][0]["status"] == "pending"

    # 2. The investigation is retrievable afterward.
    fetched = client.get(f"/investigations/{tier1_transaction_id}")
    assert fetched.status_code == 200
    assert fetched.json()["transaction_id"] == tier1_transaction_id

    # 3. The proposal shows up in the pending-approvals queue.
    approvals = client.get("/approvals").json()
    matching = [a for a in approvals if a["transaction_id"] == tier1_transaction_id]
    assert len(matching) >= 1
    approval_id = matching[0]["approval_id"]

    # No case exists yet -- consequential action is still gated.
    cases_before = client.get("/cases").json()
    assert not any(c["approval_id"] == approval_id for c in cases_before)

    # 4. A human approves it.
    decision = client.post(
        f"/approvals/{approval_id}/decision",
        json={"approved": True, "reviewer": "analyst_jane", "notes": "confirmed against evidence"},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "approved"

    # 5. Only now does the case exist.
    cases_after = client.get("/cases").json()
    created = [c for c in cases_after if c["approval_id"] == approval_id]
    assert len(created) == 1
    assert created[0]["transaction_id"] == tier1_transaction_id

    case_id = created[0]["case_id"]
    fetched_case = client.get(f"/cases/{case_id}")
    assert fetched_case.status_code == 200


def test_get_investigation_404_before_any_run(client):
    response = client.get("/investigations/never-investigated-id")
    assert response.status_code == 404


def test_decide_unknown_approval_404(client):
    response = client.post(
        "/approvals/apr_does_not_exist/decision", json={"approved": True, "reviewer": "a"}
    )
    assert response.status_code == 404


def test_decide_same_approval_twice_409(client, tier1_transaction_id):
    txn_id = tier1_transaction_id
    response = client.post("/investigations", json={"transaction_id": txn_id})
    approval_id = response.json()["pending_approvals"][0]["approval_id"]

    first = client.post(f"/approvals/{approval_id}/decision", json={"approved": True, "reviewer": "a"})
    assert first.status_code == 200
    second = client.post(f"/approvals/{approval_id}/decision", json={"approved": False, "reviewer": "b"})
    assert second.status_code == 409
