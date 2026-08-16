"""Phase 5 security regression tests: run every case declared in
security/redteam_cases.yaml and assert it's blocked.

This is a thin pytest wrapper around scripts/run_redteam.py's own engine
(bankshield.security.redteam_engine) so the same 24 deterministic,
offline attack cases that back reports/phase5_redteam_report.md also run
in CI as ordinary regression tests -- a future change that reopens one of
these gaps fails the test suite, not just the standalone script."""

from __future__ import annotations

import yaml
import pytest

from bankshield import config
from bankshield.security.redteam_engine import run_case


def _load_case_ids() -> list[str]:
    raw = yaml.safe_load(config.REDTEAM_CASES_YAML.read_text())
    return [case["attack_id"] for case in raw["cases"]]


CASE_IDS = _load_case_ids()


def test_redteam_yaml_declares_all_eight_categories():
    raw = yaml.safe_load(config.REDTEAM_CASES_YAML.read_text())
    categories = {case["category"] for case in raw["cases"]}
    assert categories == {
        "prompt_injection",
        "rag_poisoning",
        "unauthorized_tool_use",
        "approval_bypass",
        "citation_integrity",
        "information_leakage",
        "adversarial_api_input",
        "privilege_escalation",
    }


def test_redteam_yaml_cases_have_required_fields():
    raw = yaml.safe_load(config.REDTEAM_CASES_YAML.read_text())
    required = {"attack_id", "category", "description", "malicious_input", "expected_security_behavior"}
    for case in raw["cases"]:
        assert required <= case.keys(), case.get("attack_id")


def test_malicious_policy_fixture_not_in_production_docs_dir():
    """Structural guard: the red-team fixture directory and the production
    policy corpus directory must never be the same path."""
    assert config.MALICIOUS_POLICY_FIXTURES_DIR != config.POLICY_DOCS_DIR
    assert not str(config.MALICIOUS_POLICY_FIXTURES_DIR).startswith(str(config.POLICY_DOCS_DIR))


@pytest.mark.parametrize("attack_id", CASE_IDS)
def test_redteam_case_is_blocked(attack_id):
    result = run_case(attack_id)
    assert result.passed, result.actual_behavior
