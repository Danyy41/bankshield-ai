"""Authentication, role-based authorization, and the reviewer-identity
guarantees of the approval gate."""

import io
from datetime import timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from bankshield.api.app import app, get_llm_client
from bankshield.auth import cli, tokens
from bankshield.auth.passwords import hash_password, verify_password
from bankshield.auth.users import User, UserStore
from bankshield.investigation import data_access
from bankshield.investigation.approvals import get_approval_store
from bankshield.investigation.case_store import get_case_store
from bankshield.investigation.llm_client import AutoFakeLLMClient
from bankshield.persistence import audit
from bankshield.persistence.db import get_database, utcnow
from tests.conftest import TEST_PASSWORD

PROTECTED = [
    ("GET", "/auth/me"),
    ("GET", "/transactions/x"),
    ("GET", "/transactions/x/risk"),
    ("GET", "/customers/x/auth-history"),
    ("GET", "/customers/x/graph-neighbors"),
    ("POST", "/policy/search"),
    ("GET", "/demo/sample-transactions"),
    ("POST", "/investigations"),
    ("GET", "/investigations/x"),
    ("GET", "/approvals"),
    ("POST", "/approvals/apr_x/decision"),
    ("GET", "/cases"),
    ("GET", "/cases/case_x"),
    ("GET", "/audit"),
]


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[get_llm_client] = lambda: AutoFakeLLMClient()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def tier1():
    for sample in data_access.sample_transactions(n=5):
        if sample["risk_tier"] == "tier_1":
            return sample["transaction_id"]
    pytest.skip("no tier_1 sample")


def _pending_approval(client, headers, txn_id):
    body = client.post("/investigations", json={"transaction_id": txn_id}, headers=headers).json()
    return body["pending_approvals"][0]["approval_id"]


# --- login -------------------------------------------------------------------


def test_login_issues_a_token_that_works(client, test_users):
    response = client.post("/auth/token", data={"username": "alice.analyst", "password": TEST_PASSWORD})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer" and body["role"] == "analyst"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.json() == {"username": "alice.analyst", "role": "analyst"}


@pytest.mark.parametrize("username, password", [("alice.analyst", "wrong-password-123"), ("nobody.here", TEST_PASSWORD)])
def test_bad_credentials_are_rejected_without_revealing_which_part(client, test_users, username, password):
    response = client.post("/auth/token", data={"username": username, "password": password})
    assert response.status_code == 401
    assert response.json() == {"detail": "incorrect username or password"}


def test_repeated_failures_lock_the_account_temporarily(client):
    UserStore().create("lockout.target", TEST_PASSWORD, "analyst")
    for _ in range(5):
        assert client.post("/auth/token", data={"username": "lockout.target", "password": "nope-nope-nope"}).status_code == 401
    # Even the right password is refused while locked out.
    locked = client.post("/auth/token", data={"username": "lockout.target", "password": TEST_PASSWORD})
    assert locked.status_code == 429


def test_passwords_are_stored_as_salted_scrypt_hashes():
    first, second = hash_password("a-long-enough-password"), hash_password("a-long-enough-password")
    assert first.startswith("scrypt$") and first != second  # random salt
    assert verify_password("a-long-enough-password", first)
    assert not verify_password("a-long-enough-passwore", first)
    with pytest.raises(ValueError):
        hash_password("short")


# --- authentication on every protected endpoint -----------------------------------


@pytest.mark.parametrize("method, path", PROTECTED)
def test_protected_endpoints_require_a_token(client, method, path):
    assert client.request(method, path).status_code == 401


def test_public_endpoints_stay_public(client):
    assert client.get("/health").status_code == 200
    assert client.get("/a2a/triage/.well-known/agent-card.json").status_code == 200


def _forge(claims_overrides=None, key=None, algorithm="HS256", unsigned=False):
    now = utcnow()
    claims = {"sub": "rita.reviewer", "role": "reviewer", "iss": tokens.ISSUER, "aud": tokens.AUDIENCE,
              "iat": now, "exp": now + timedelta(minutes=5)}
    claims.update(claims_overrides or {})
    if unsigned:
        return jwt.encode(claims, None, algorithm="none")
    return jwt.encode(claims, key if key is not None else tokens._secret(), algorithm=algorithm)


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        _forge(unsigned=True),  # alg=none, every claim otherwise valid
        _forge(key="attacker-controlled-key-" + "k" * 20),
        _forge({"exp": utcnow() - timedelta(minutes=1)}),
        _forge({"iss": "someone-else"}),
        _forge({"aud": "another-api"}),
        _forge({"sub": "ghost.user"}),
    ],
    ids=["garbage", "alg-none", "wrong-key", "expired", "wrong-issuer", "wrong-audience", "unknown-user"],
)
def test_invalid_tokens_are_rejected(client, test_users, token):
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_disabling_a_user_revokes_their_existing_token_immediately(client):
    store = UserStore()
    user = store.create("soon.disabled", TEST_PASSWORD, "reviewer")
    headers = {"Authorization": f"Bearer {tokens.issue(user)}"}
    assert client.get("/auth/me", headers=headers).status_code == 200
    store.set_disabled("soon.disabled", True)
    assert client.get("/auth/me", headers=headers).status_code == 401


def test_role_comes_from_the_database_not_the_token(client):
    """A token minted while the user was a reviewer stops granting reviewer
    powers as soon as the role changes."""
    from sqlalchemy import update

    from bankshield.persistence.db import users

    store = UserStore()
    store.create("demoted.reviewer", TEST_PASSWORD, "reviewer")
    token = tokens.issue(User("demoted.reviewer", "reviewer", False))
    with get_database().transaction() as conn:
        conn.execute(update(users).where(users.c.username == "demoted.reviewer").values(role="analyst"))
    response = client.post("/approvals/apr_x/decision", json={"approved": True}, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


# --- role-based authorization -------------------------------------------------------


@pytest.mark.parametrize("role, expected", [("analyst", 403), ("admin", 403), ("agent", 403), ("reviewer", 404)])
def test_only_reviewers_may_decide_approvals(client, auth_headers, role, expected):
    # 404 for the reviewer = authorized, and the approval simply doesn't exist.
    response = client.post("/approvals/apr_missing/decision", json={"approved": True}, headers=auth_headers(role))
    assert response.status_code == expected


@pytest.mark.parametrize("role, expected", [("analyst", 403), ("reviewer", 403), ("agent", 403), ("admin", 200)])
def test_only_admins_read_the_audit_log(client, auth_headers, role, expected):
    assert client.get("/audit", headers=auth_headers(role)).status_code == expected


def test_agents_cannot_run_investigations_over_http(client, auth_headers, tier1):
    # Service accounts drive investigations through the A2A agents only.
    response = client.post("/investigations", json={"transaction_id": tier1}, headers=auth_headers("agent"))
    assert response.status_code == 403


def test_denials_are_audited(client, auth_headers):
    client.post("/approvals/apr_missing/decision", json={"approved": True}, headers=auth_headers("analyst"))
    events = audit.recent(get_database(), limit=20)
    assert any(e["action"] == "access.denied" and e["actor"] == "alice.analyst" for e in events)


# --- reviewer identity and segregation of duties ----------------------------------------


def test_decision_is_attributed_to_the_authenticated_reviewer(client, auth_headers, tier1):
    approval_id = _pending_approval(client, auth_headers("analyst"), tier1)
    assert get_approval_store().get(approval_id).requested_by == "alice.analyst"
    decided = client.post(f"/approvals/{approval_id}/decision", json={"approved": True}, headers=auth_headers("reviewer"))
    assert decided.status_code == 200 and decided.json()["reviewer"] == "rita.reviewer"
    (case,) = [c for c in get_case_store().list_all() if c.approval_id == approval_id]
    assert case.approved_by == "rita.reviewer"


def test_reviewers_cannot_decide_in_someone_elses_name(client, auth_headers, tier1):
    approval_id = _pending_approval(client, auth_headers("analyst"), tier1)
    response = client.post(
        f"/approvals/{approval_id}/decision",
        json={"approved": True, "reviewer": "rob.reviewer"},
        headers=auth_headers("reviewer"),
    )
    assert response.status_code == 403
    assert get_approval_store().get(approval_id).status.value == "pending"
    # Naming yourself is fine.
    ok = client.post(
        f"/approvals/{approval_id}/decision",
        json={"approved": False, "reviewer": "rita.reviewer"},
        headers=auth_headers("reviewer"),
    )
    assert ok.status_code == 200 and ok.json()["status"] == "rejected"


def test_reviewers_cannot_approve_their_own_proposals(client, auth_headers, tier1):
    approval_id = _pending_approval(client, auth_headers("reviewer"), tier1)
    own = client.post(f"/approvals/{approval_id}/decision", json={"approved": True}, headers=auth_headers("reviewer"))
    assert own.status_code == 403
    assert not [c for c in get_case_store().list_all() if c.approval_id == approval_id]
    other = client.post(f"/approvals/{approval_id}/decision", json={"approved": True}, headers=auth_headers("reviewer2"))
    assert other.status_code == 200


# --- account provisioning -----------------------------------------------------------------


def test_cli_creates_and_disables_users(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("cli-created-password\n"))
    cli.main(["create-user", "cli.reviewer", "--role", "reviewer", "--password-stdin"])
    assert UserStore().authenticate("cli.reviewer", "cli-created-password").role == "reviewer"
    cli.main(["disable-user", "cli.reviewer"])
    assert UserStore().authenticate("cli.reviewer", "cli-created-password") is None
    with pytest.raises(SystemExit):
        cli.main(["create-user", "cli.badrole", "--role", "superuser", "--password-stdin"])
