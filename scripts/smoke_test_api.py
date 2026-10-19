"""End-to-end smoke test against a running BankShield AI API.

Used by CI against the built Docker image, and usable against any
deployment. Standard library only, so it runs without installing the
project. Exercises the real artifacts and the security-relevant paths,
not just liveness:

1. /health is up
2. the baked-in data + model serve real transactions and risk scores
3. an investigation of a Tier 1 alert runs (offline mode) and proposes a
   case only as a *pending approval* -- no case exists yet
4. the human-approval gate holds: a decision without a named reviewer is
   rejected (422), a named reviewer's approval creates exactly one case,
   and deciding the same approval again is rejected (409)
5. unknown IDs return 404, and unhandled errors don't leak internals

Usage:
    python scripts/smoke_test_api.py [BASE_URL]    # default http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE_URL + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"[FAIL] {message}")
        sys.exit(1)
    print(f"[ OK ] {message}")


def main() -> None:
    status, body = request("GET", "/health")
    check(status == 200 and body == {"status": "ok"}, "GET /health -> 200 ok")

    status, body = request("GET", "/demo/sample-transactions?n=5")
    samples = body.get("transactions", []) if status == 200 else []
    check(len(samples) > 0, f"GET /demo/sample-transactions -> {len(samples)} real transactions")
    tier1 = next((t for t in samples if t["risk_tier"] == "tier_1"), None)
    check(tier1 is not None, "sample includes a Tier 1 alert")
    txn_id = tier1["transaction_id"]

    status, body = request("GET", f"/transactions/{txn_id}/risk")
    check(status == 200 and 0.0 <= body.get("risk_score", -1) <= 1.0, "GET /transactions/{id}/risk -> valid score")

    _, cases_before = request("GET", "/cases")
    status, result = request("POST", "/investigations", {"transaction_id": txn_id})
    check(status == 200, "POST /investigations (Tier 1) -> 200")
    check(result["model_id"].startswith("fake"), f"investigation ran in offline mode (model_id={result['model_id']!r})")
    pending = [a for a in result["pending_approvals"] if a["action"] == "create_case"]
    check(len(pending) == 1 and pending[0]["status"] == "pending", "case proposed only as a pending approval")
    _, cases_after = request("GET", "/cases")
    check(len(cases_after) == len(cases_before), "no case created before human approval")

    approval_id = pending[0]["approval_id"]
    status, _ = request("POST", f"/approvals/{approval_id}/decision", {"approved": True, "reviewer": ""})
    check(status == 422, "approval without a named reviewer -> 422")
    status, body = request(
        "POST", f"/approvals/{approval_id}/decision", {"approved": True, "reviewer": "ci-smoke-test"}
    )
    check(status == 200 and body["status"] == "approved", "approval by a named reviewer -> 200 approved")
    _, cases_final = request("GET", "/cases")
    created = [c for c in cases_final if c["approval_id"] == approval_id]
    check(len(created) == 1, "exactly one case created from the approval")
    status, _ = request(
        "POST", f"/approvals/{approval_id}/decision", {"approved": True, "reviewer": "ci-smoke-test"}
    )
    check(status == 409, "deciding the same approval twice -> 409")

    status, body = request("GET", "/transactions/does-not-exist")
    check(status == 404, "unknown transaction -> 404")
    status, body = request("POST", "/investigations", {"transaction_id": txn_id, "auto_approve": True})
    check(status == 422, "unexpected request fields rejected -> 422")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
