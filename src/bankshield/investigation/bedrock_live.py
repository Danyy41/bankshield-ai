"""Helpers for running BankShield against live Amazon Bedrock: a free
preflight (no inference), redaction of AWS identifiers, and the checks a
live investigation must pass before its output is published as an example.

Used by scripts/bedrock_preflight.py and
scripts/run_live_bedrock_investigation.py. Nothing here runs in the test
suite against AWS -- the tests pass stub boto3 clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from bankshield.investigation.agent import CITATION_PATTERN
from bankshield.investigation.llm_client import bedrock_settings
from bankshield.investigation.schemas import InvestigationResult

# --- redaction ----------------------------------------------------------------

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\barn:aws[a-z-]*:[^\s\"',]+"), "<redacted-arn>"),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "<redacted-access-key>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<redacted-uuid>"),
    (re.compile(r"(?<!\d)\d{12}(?!\d)"), "<redacted-account-id>"),
]


def redact(value: Any) -> Any:
    """Recursively redact ARNs, access key IDs, request/approval UUIDs and
    12-digit account IDs from strings in a JSON-like structure."""
    if isinstance(value, str):
        for pattern, replacement in _REDACTIONS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


# --- preflight ----------------------------------------------------------------


@dataclass
class PreflightReport:
    model_id: str
    region: str
    ok: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.blockers


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return response.get("Error", {}).get("Code", type(exc).__name__)


def _base_model_id(model_id: str) -> str:
    """us.anthropic.claude-opus-5-5 -> anthropic.claude-opus-5-5"""
    head, _, rest = model_id.partition(".")
    return rest if head in {"us", "eu", "jp", "au", "apac", "global"} and rest else model_id


def run_preflight(client_factory: Callable[..., Any] | None = None) -> PreflightReport:
    """Check everything a live investigation needs using only free API calls
    (sts:GetCallerIdentity, bedrock:GetInferenceProfile,
    bedrock:GetFoundationModelAvailability). Never calls the model.

    `client_factory(service, region_name=...)` defaults to boto3.client;
    tests inject stubs."""
    import boto3

    model_id, region = bedrock_settings()
    report = PreflightReport(model_id=model_id, region=region)
    make = client_factory or boto3.client

    if client_factory is None:
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            report.blockers.append(
                "No AWS credentials found. Configure one standard source: `aws configure sso` / "
                "`aws sso login --profile P` and export AWS_PROFILE=P, an IAM role (ECS task role, "
                "EC2 instance profile), or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY[/AWS_SESSION_TOKEN]."
            )
            return report
        report.ok.append(f"credentials found (source: {credentials.method})")

    try:
        identity = make("sts", region_name=region).get_caller_identity()
        principal_kind = identity.get("Arn", "").split(":")[5].split("/")[0] if identity.get("Arn") else "?"
        report.ok.append(f"credentials valid (account ****{identity.get('Account', '')[-4:]}, {principal_kind})")
    except Exception as exc:  # noqa: BLE001 -- report, don't crash
        report.blockers.append(
            f"AWS rejected the credentials ({_error_code(exc)}). Refresh them (e.g. `aws sso login`) "
            "or replace the access key; for temporary keys also set AWS_SESSION_TOKEN."
        )
        return report

    bedrock = make("bedrock", region_name=region)
    try:
        profile = bedrock.get_inference_profile(inferenceProfileIdentifier=model_id)
        regions = sorted({m.get("modelArn", "").split(":")[3] for m in profile.get("models", [])} - {""})
        report.ok.append(f"inference profile {model_id} is {profile.get('status')} in {region} (routes to {', '.join(regions) or '?'})")
    except Exception as exc:  # noqa: BLE001
        code = _error_code(exc)
        if code == "AccessDeniedException":
            report.warnings.append(
                "Could not read the inference profile (no bedrock:GetInferenceProfile) -- optional, "
                "only the preflight needs it."
            )
        elif code in {"ResourceNotFoundException", "ValidationException"}:
            report.blockers.append(
                f"Inference profile {model_id} does not exist in {region}. Use a region that serves it "
                "(US profile: us-east-1, us-east-2, us-west-2) or set BANKSHIELD_BEDROCK_MODEL_ID to the "
                "profile for your geography (eu./jp./au./global.anthropic.claude-opus-5-5)."
            )
        else:
            report.blockers.append(f"Bedrock control plane unreachable in {region} ({code}).")

    base_model = _base_model_id(model_id)
    try:
        availability = bedrock.get_foundation_model_availability(modelId=base_model)
        problems = []
        if availability.get("authorizationStatus") != "AUTHORIZED":
            problems.append("not authorized for this account (IAM/SCP or model access)")
        if availability.get("entitlementAvailability") != "AVAILABLE":
            problems.append("no entitlement -- enable model access in the Bedrock console")
        agreement = availability.get("agreementAvailability", {}).get("status")
        if agreement != "AVAILABLE":
            problems.append(
                f"model agreement {agreement or 'missing'} -- submit Anthropic's first-time-use form / "
                "accept the offer in the Bedrock console (Model catalog)"
            )
        if availability.get("regionAvailability") != "AVAILABLE":
            problems.append(f"not available in {region}")
        if problems:
            report.blockers.append(f"{base_model}: " + "; ".join(problems))
        else:
            report.ok.append(f"model access to {base_model} is enabled")
    except Exception as exc:  # noqa: BLE001
        if _error_code(exc) == "AccessDeniedException":
            report.warnings.append(
                "Could not check model access (no bedrock:GetFoundationModelAvailability) -- verify it "
                "in the Bedrock console instead."
            )
        else:
            report.warnings.append(f"Model-access check failed ({_error_code(exc)}); verify it in the console.")

    report.warnings.append(
        "bedrock:InvokeModel itself can't be verified without a (billable) call -- see "
        "deploy/iam/bedrock-invoke-policy.json."
    )
    return report


# --- live-run verification ------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def verify_live_result(result: InvestigationResult, cases_before: int, cases_after: int) -> list[Check]:
    """The properties a live investigation must show before it is published:
    the agent actually used the tools, every citation it made resolves to a
    passage it retrieved, guardrails held, and a proposed case is still only
    pending human approval."""
    tool_names = [c.tool_name for c in result.tool_calls]
    tool_errors = [c.tool_name for c in result.tool_calls if c.is_error]
    mentioned = {f"{d} {s}" for d, s in CITATION_PATTERN.findall(result.narrative)}
    verified = {f"{c.doc_id} {c.section}" for c in result.citations}
    proposed = [a for a in result.pending_approvals if a.action == "create_case"]

    checks = [
        Check("real model", not result.model_id.startswith("fake"), result.model_id),
        Check("tools used", bool(tool_names), ", ".join(tool_names) or "none"),
        Check("no tool errors", not tool_errors, ", ".join(tool_errors) or "none"),
        Check("cited", bool(verified), f"{len(verified)} verified citation(s)"),
        Check("no unverified citations", mentioned <= verified,
              ", ".join(sorted(mentioned - verified)) or "none"),
        Check("approval gate held", cases_after == cases_before and all(a.status == "pending" for a in proposed),
              f"{len(proposed)} pending proposal(s), cases {cases_before} -> {cases_after}"),
    ]
    return checks
