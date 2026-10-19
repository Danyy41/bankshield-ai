"""The Bedrock access diagnosis (bedrock_diagnostics.py) against stubbed
AWS clients: classification of AWS's AccessDenied messages, sanitizing,
and the end-to-end diagnosis. No network calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import boto3
import pytest
from botocore.stub import ANY, Stubber

from bankshield.investigation.bedrock_diagnostics import classify_access_denied, diagnose, sanitize

ACCOUNT = "123456789012"
SSO_ROLE = "AWSReservedSSO_BankShieldBedrock_0123456789abcdef"
CALLER = f"arn:aws:sts::{ACCOUNT}:assumed-role/{SSO_ROLE}/alice@example.com"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/aws-reserved/sso.amazonaws.com/{SSO_ROLE}"
PROFILE_ID = "us.anthropic.claude-opus-5-5"
PROFILE_ARN = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:inference-profile/{PROFILE_ID}"
REGIONS = ("us-east-1", "us-east-2", "us-west-2")
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)

SCP_DENY = (
    f"User: {CALLER} is not authorized to perform: bedrock:InvokeModel on resource: "
    "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-opus-5-5 "
    "with an explicit deny in a service control policy"
)


@pytest.mark.parametrize(
    ("message", "category"),
    [
        (SCP_DENY, "scp"),
        ("... on resource: X with an explicit deny in a resource control policy", "rcp"),
        ("... because no permissions boundary allows the bedrock:InvokeModel action", "permissions_boundary"),
        ("... with an explicit deny in a session policy", "session_policy"),
        ("... because no VPC endpoint policy allows the bedrock:InvokeModel action", "vpc_endpoint_policy"),
        ("... with an explicit deny in an identity-based policy", "identity_explicit_deny"),
        ("... because no identity-based policy allows the bedrock:InvokeModel action", "identity_no_allow"),
        ("Model access is denied due to IAM user or service role is not authorized to perform the required "
         "AWS Marketplace actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)", "marketplace"),
        ("Model use case details have not been submitted for this account.", "use_case_form"),
        ("You don't have access to the model with the specified model ID.", "model_access"),
        ("Something else entirely", "unknown"),
    ],
)
def test_classify_access_denied(message, category):
    assert classify_access_denied(message).category == category


def test_sanitize_masks_account_and_session_but_keeps_request_id():
    text = sanitize(f"{SCP_DENY} request 1b4e28ba-2fa1-11d2-883f-0016d3cca427")
    assert ACCOUNT not in text and "alice@example.com" not in text
    assert "****9012" in text and "<session>" in text
    assert "1b4e28ba-2fa1-11d2-883f-0016d3cca427" in text and "service control policy" in text


class _Clients:
    """client_factory for diagnose(): one stubbed client per (service, region)."""

    def __init__(self):
        self.clients: dict[tuple[str, str], object] = {}
        self.stubs: dict[tuple[str, str], Stubber] = {}

    def __call__(self, service, region_name):
        key = (service, region_name)
        if key not in self.clients:
            client = boto3.client(service, region_name=region_name,
                                  aws_access_key_id="testing", aws_secret_access_key="testing")
            self.clients[key], self.stubs[key] = client, Stubber(client)
            self.stubs[key].activate()
        return self.clients[key]

    def stub(self, service, region="us-east-1"):
        self(service, region)
        return self.stubs[(service, region)]


def _sim(decision="allowed", org=None):
    result = {"EvalActionName": "bedrock:InvokeModel", "EvalDecision": decision,
              "MatchedStatements": [{"SourcePolicyId": "BankShieldBedrockInvoke"}] if decision == "allowed" else []}
    if org is not None:
        result["OrganizationsDecisionDetail"] = {"AllowedByOrganizations": org}
    return {"EvaluationResults": [result]}


def _queue(c: _Clients, *, scp_denies_region=None, trail_message=None, agreement="AVAILABLE"):
    c.stub("sts").add_response("get_caller_identity", {"Account": ACCOUNT, "Arn": CALLER, "UserId": "AROA:x"})
    iam = c.stub("iam")
    iam.add_response("get_role", {"Role": {"Path": "/aws-reserved/sso.amazonaws.com/", "RoleName": SSO_ROLE,
                                           "RoleId": "AROAEXAMPLEEXAMPLE01", "Arn": ROLE_ARN,
                                           "CreateDate": NOW}}, {"RoleName": SSO_ROLE})
    bedrock = c.stub("bedrock")
    bedrock.add_response("get_foundation_model_availability", {
        "modelId": "anthropic.claude-opus-5-5", "agreementAvailability": {"status": agreement},
        "authorizationStatus": "AUTHORIZED", "entitlementAvailability": "AVAILABLE", "regionAvailability": "AVAILABLE",
    })
    bedrock.add_response("get_use_case_for_model_access", {"formData": b"{}"})
    bedrock.add_response("get_inference_profile", {
        "inferenceProfileName": "US Claude Opus 5.5", "inferenceProfileArn": PROFILE_ARN,
        "inferenceProfileId": PROFILE_ID, "status": "ACTIVE", "type": "SYSTEM_DEFINED",
        "models": [{"modelArn": f"arn:aws:bedrock:{r}::foundation-model/anthropic.claude-opus-5-5"} for r in REGIONS],
    })
    # profile, one foundation model per destination region, two Marketplace actions
    iam.add_response("simulate_principal_policy", _sim(org=True),
                     {"PolicySourceArn": ROLE_ARN, "ActionNames": ["bedrock:InvokeModel"],
                      "ResourceArns": [PROFILE_ARN], "ContextEntries": ANY})
    for r in REGIONS:
        iam.add_response("simulate_principal_policy", _sim(org=(r != scp_denies_region)), {
            "PolicySourceArn": ROLE_ARN, "ActionNames": ["bedrock:InvokeModel"],
            "ResourceArns": [f"arn:aws:bedrock:{r}::foundation-model/anthropic.claude-opus-5-5"],
            "ContextEntries": [
                {"ContextKeyName": "bedrock:InferenceProfileArn", "ContextKeyValues": [PROFILE_ARN],
                 "ContextKeyType": "string"},
                {"ContextKeyName": "aws:RequestedRegion", "ContextKeyValues": [r], "ContextKeyType": "string"},
            ],
        })
    for _ in range(2):
        iam.add_response("simulate_principal_policy", _sim("implicitDeny"))
    for r in REGIONS:
        events = []
        if trail_message and r == "us-east-1":
            events.append({"EventId": "e1", "CloudTrailEvent": json.dumps({
                "eventName": "Converse", "eventTime": "2026-10-19T11:50:00Z", "awsRegion": r,
                "errorCode": "AccessDenied", "errorMessage": trail_message,
                "requestID": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
            })})
        c.stub("cloudtrail", r).add_response("lookup_events", {"Events": events})


def test_diagnosis_finds_scp_from_cloudtrail_and_simulation():
    c = _Clients()
    _queue(c, scp_denies_region="us-west-2", trail_message=SCP_DENY)
    d = diagnose(PROFILE_ID, "us-east-1", c, now=NOW)

    assert [cause.category for cause, _ in d.causes] == ["scp"]
    evidence = d.causes[0][1]
    assert evidence.startswith("CloudTrail 1b4e28ba") and "simulation: SCPs deny" in evidence
    text = "\n".join(line for _, lines in d.sections for line in lines)
    assert "routes to: us-east-1, us-east-2, us-west-2" in text
    assert "SCPs allow: False" in text and "1b4e28ba-2fa1-11d2-883f-0016d3cca427" in text
    assert ACCOUNT not in text and "alice@example.com" not in text
    for stub in c.stubs.values():
        stub.assert_no_pending_responses()


def test_diagnosis_reports_agreement_and_clean_iam():
    c = _Clients()
    _queue(c, agreement="PENDING")
    d = diagnose(PROFILE_ID, "us-east-1", c, now=NOW)
    assert [cause.category for cause, _ in d.causes] == ["model_access"]


def test_diagnosis_stops_on_invalid_credentials():
    c = _Clients()
    c.stub("sts").add_client_error("get_caller_identity", service_error_code="InvalidClientTokenId",
                                   http_status_code=403)
    d = diagnose(PROFILE_ID, "us-east-1", c, now=NOW)
    assert not d.credentials_ok and not d.causes


# --- reading one denial message (the four buckets) ------------------------------------

from bankshield.investigation.bedrock_diagnostics import (  # noqa: E402
    BUCKET_ACCOUNT,
    BUCKET_IAM,
    BUCKET_MARKETPLACE,
    BUCKET_OTHER,
    explain_denial,
    reproduce_converse,
)

USER = f"arn:aws:iam::{ACCOUNT}:user/bankshield-bedrock"
FM = "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-5-5"
NO_ALLOW = "because no identity-based policy allows the bedrock:InvokeModel action"
MARKETPLACE = (
    "Model access is denied due to IAM user or service role is not authorized to perform the required AWS "
    "Marketplace actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe) to enable access to "
    "this model. Refer to the Amazon Bedrock documentation for further details. Your AWS Marketplace "
    "subscription for this model cannot be completed at this time. If you recently fixed this issue, try "
    "again after 15 minutes."
)


@pytest.mark.parametrize(
    ("message", "bucket", "category", "action", "resource_type"),
    [
        (MARKETPLACE, BUCKET_MARKETPLACE, "marketplace", None, None),
        (f"User: {USER} is not authorized to perform: aws-marketplace:Subscribe on resource: * "
         "because no identity-based policy allows the aws-marketplace:Subscribe action",
         BUCKET_MARKETPLACE, "marketplace", "aws-marketplace:Subscribe", "other"),
        (f"User: {USER} is not authorized to perform: bedrock:InvokeModel on resource: {PROFILE_ARN} {NO_ALLOW}",
         BUCKET_IAM, "identity_no_allow", "bedrock:InvokeModel", "inference-profile"),
        (f"User: {USER} is not authorized to perform: bedrock:InvokeModel on resource: {FM} {NO_ALLOW}",
         BUCKET_IAM, "identity_no_allow", "bedrock:InvokeModel", "foundation-model"),
        (f"User: {USER} is not authorized to perform: bedrock:InvokeModel on resource: {FM} "
         "with an explicit deny in a service control policy",
         BUCKET_ACCOUNT, "scp", "bedrock:InvokeModel", "foundation-model"),
        ("You don't have access to the model with the specified model ID.", BUCKET_ACCOUNT, "model_access", None, None),
        ("anthropic.claude-opus-5-5 is not available for this account.", BUCKET_ACCOUNT, "model_not_offered",
         None, None),
        ("Model use case details have not been submitted for this account.", BUCKET_ACCOUNT, "use_case_form", None, None),
        (f"User: {USER} is not authorized to perform: kms:Decrypt on resource: "
         f"arn:aws:kms:us-east-1:{ACCOUNT}:key/k because no identity-based policy allows the kms:Decrypt action",
         BUCKET_OTHER, "identity_no_allow", "kms:Decrypt", "other"),
    ],
)
def test_explain_denial_buckets(message, bucket, category, action, resource_type):
    e = explain_denial(message)
    assert (e.bucket, e.cause.category, e.action, e.resource_type) == (bucket, category, action, resource_type)
    assert e.resource is None or ACCOUNT not in e.resource


def _runtime():
    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          aws_access_key_id="testing", aws_secret_access_key="testing")
    return client, Stubber(client)


def test_reproduce_converse_reports_sanitized_denial_with_request_id():
    client, stub = _runtime()
    message = f"User: {USER} is not authorized to perform: bedrock:InvokeModel on resource: {FM} {NO_ALLOW}"
    stub.add_client_error(
        "converse", service_error_code="AccessDeniedException", service_message=message, http_status_code=403,
        response_meta={"RequestId": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
                       "HTTPHeaders": {"x-amzn-errortype": "AccessDeniedException:http://internal.amazon.com/"}},
        expected_params={"modelId": PROFILE_ID, "messages": [{"role": "user", "content": [{"text": "ping"}]}],
                         "inferenceConfig": {"maxTokens": 1}},
    )
    with stub:
        r = reproduce_converse(client, PROFILE_ID)
    stub.assert_no_pending_responses()
    assert r["outcome"] == "denied" and r["code"] == "AccessDeniedException" and r["http_status"] == 403
    assert r["request_id"] == "1b4e28ba-2fa1-11d2-883f-0016d3cca427"
    assert ACCOUNT not in r["message"] and "****9012" in r["message"]
    assert r["explanation"].bucket == BUCKET_IAM and r["explanation"].resource_type == "foundation-model"


def test_reproduce_converse_validation_error_means_authorized():
    client, stub = _runtime()
    stub.add_client_error("converse", service_error_code="ValidationException", http_status_code=400)
    with stub:
        assert reproduce_converse(client, PROFILE_ID)["outcome"] == "authorized"


def test_reproduce_converse_allowed():
    client, stub = _runtime()
    stub.add_response("converse", {
        "output": {"message": {"role": "assistant", "content": [{"text": "p"}]}}, "stopReason": "max_tokens",
        "usage": {"inputTokens": 8, "outputTokens": 1, "totalTokens": 9}, "metrics": {"latencyMs": 1},
    })
    with stub:
        r = reproduce_converse(client, PROFILE_ID)
    assert r["outcome"] == "allowed" and r["usage"]["outputTokens"] == 1
