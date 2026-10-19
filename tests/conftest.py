import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Every test run gets its own throwaway database and signing key -- set
# before anything imports the app, so tests never touch var/bankshield.db.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="bankshield-tests-")
os.environ["BANKSHIELD_DATABASE_URL"] = f"sqlite:///{_TEST_DB_DIR}/bankshield.db"
os.environ["BANKSHIELD_JWT_SECRET"] = "test-only-signing-key-" + "x" * 32

import pytest  # noqa: E402

TEST_PASSWORD = "correct-horse-battery-staple"
# fixture key -> (username, role)
TEST_USERS = {
    "analyst": ("alice.analyst", "analyst"),
    "reviewer": ("rita.reviewer", "reviewer"),
    "reviewer2": ("rob.reviewer", "reviewer"),
    "admin": ("ada.admin", "admin"),
    "agent": ("svc-triage", "agent"),
}


@pytest.fixture(scope="session")
def test_users():
    from bankshield.auth.users import UserStore

    store = UserStore()
    for username, role in TEST_USERS.values():
        if store.get(username) is None:
            store.create(username, TEST_PASSWORD, role)
    return {key: store.get(username) for key, (username, _) in TEST_USERS.items()}


@pytest.fixture(scope="session")
def auth_headers(test_users):
    """auth_headers("reviewer") -> {"Authorization": "Bearer <token>"}"""
    from bankshield.auth import tokens

    issued = {key: tokens.issue(user) for key, user in test_users.items()}
    return lambda key: {"Authorization": f"Bearer {issued[key]}"}
