"""Explain *why* Bedrock denies an InvokeModel/Converse call, using only
free, read-only AWS APIs -- never the model itself.

AWS's AccessDeniedException message names the policy type that decided
the request ("... with an explicit deny in a service control policy",
"... because no identity-based policy allows ...") or the Bedrock-side
gate (model access, Marketplace subscription, use-case form). The
diagnosis collects that message from CloudTrail (the denied call itself),
plus the evidence around it:

1. who is calling (STS), and the IAM role behind an SSO session
2. model access, agreement and the Anthropic use-case form (Bedrock)
3. which regions the inference profile routes to (Bedrock)
4. an IAM simulation of exactly what the request evaluates -- the profile
   in the source region, and the foundation model in *each* destination
   region with the `bedrock:InferenceProfileArn` and `aws:RequestedRegion`
   context Bedrock supplies -- including the Organizations (SCP) and
   permissions-boundary verdicts when the simulator reports them
5. Marketplace permissions, which the first call of a third-party model
   in an account needs if the model isn't subscribed yet

Note there is no separate IAM action for Converse: it is authorized as
`bedrock:InvokeModel` (ConverseStream as InvokeModelWithResponseStream).

Output is sanitized for local display: account IDs are shown as their last
four digits and SSO session names (usually an email) are hidden. Request
IDs are kept -- they identify the failed call for AWS Support.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

RUNTIME_EVENTS = {"InvokeModel", "InvokeModelWithResponseStream", "Converse", "ConverseStream"}

# --- sanitizing -------------------------------------------------------------------

_ACCOUNT_RE = re.compile(r"(?<!\d)(\d{8})(\d{4})(?!\d)")
_SESSION_RE = re.compile(r"(assumed-role/[^/\s\"',]+/)[^\s\"',]+")


def sanitize(text: str) -> str:
    """Mask account IDs to their last four digits and hide role session
    names; keep everything else (region, model, policy type) readable."""
    text = _SESSION_RE.sub(r"\1<session>", text)
    return _ACCOUNT_RE.sub(lambda m: "****" + m.group(2), text)


# --- classifying AWS's own explanation ---------------------------------------------


@dataclass(frozen=True)
class Cause:
    category: str
    explanation: str
    fix: str


_CAUSES: list[tuple[tuple[str, ...], Cause]] = [
    (("service control policy",), Cause(
        "scp",
        "An AWS Organizations service control policy explicitly denies the call.",
        "Ask the organization admin which SCP denies bedrock:InvokeModel. Region-restriction SCPs "
        "(aws:RequestedRegion) must allow every region the inference profile routes to, and "
        "Bedrock-specific SCPs must allow the model/profile. The BankShield policy itself is fine.",
    )),
    (("resource control policy",), Cause(
        "rcp", "An AWS Organizations resource control policy explicitly denies the call.",
        "Ask the organization admin to review the RCP applied to this account.",
    )),
    (("permissions boundary",), Cause(
        "permissions_boundary", "The role's permissions boundary doesn't allow bedrock:InvokeModel here.",
        "Add bedrock:InvokeModel on the profile and the foundation model to the boundary policy "
        "(the boundary must allow it too; it only ever narrows).",
    )),
    (("session policy",), Cause(
        "session_policy", "A session policy passed when the role was assumed denies or omits the call.",
        "Check what assumes the role (SSO permission set, AssumeRole Policy parameter) and its session policy.",
    )),
    (("vpc endpoint policy",), Cause(
        "vpc_endpoint_policy", "A VPC endpoint policy on the bedrock-runtime endpoint denies the call.",
        "Allow bedrock:InvokeModel for this role in the endpoint policy, or call outside that VPC.",
    )),
    (("explicit deny in an identity-based policy",), Cause(
        "identity_explicit_deny", "Another policy attached to the role explicitly denies the call.",
        "Find the Deny statement on the role (or its groups/permission set) covering bedrock:InvokeModel.",
    )),
    (("no identity-based policy allows",), Cause(
        "identity_no_allow",
        "No statement in the role's policies matched this exact request.",
        "Compare the resource in the message with the policy. For a foundation-model ARN, check the "
        "region is covered and the bedrock:InferenceProfileArn condition equals the profile ARN exactly "
        "(profile region and account). Fix the statement; don't widen it to bedrock:*.",
    )),
    (("aws-marketplace", "marketplace subscription"), Cause(
        "marketplace",
        "The model isn't subscribed in this account yet, and the caller can't subscribe on first use.",
        "Have an admin enable the model once (Bedrock console > Model catalog > request access, which "
        "subscribes the account), instead of granting aws-marketplace:Subscribe to the app role.",
    )),
    (("use case",), Cause(
        "use_case_form", "Anthropic's one-time use-case form hasn't been submitted for this account.",
        "Submit it in the Bedrock console (Model catalog > an Anthropic model > request access).",
    )),
    (("is not available for this account",), Cause(
        "model_not_offered",
        "Bedrock doesn't offer this model to this account (even if IAM allows it and the agreement shows "
        "AVAILABLE) -- an account-level availability restriction, not a permissions problem.",
        "Use a model the account can invoke (set BANKSHIELD_BEDROCK_MODEL_ID; check with "
        "scripts/bedrock_model_access_test.py), or ask AWS (account team / Support, citing the request "
        "ID) to enable this model for the account. Changing IAM won't help.",
    )),
    (("don't have access to the model", "do not have access to the model", "not have access to the model"), Cause(
        "model_access", "Model access for this model isn't granted/accepted in this account and region.",
        "Enable access in the Bedrock console (Model catalog) in the source region, and accept the "
        "model agreement if one is offered.",
    )),
]


def classify_access_denied(message: str) -> Cause:
    lowered = message.lower()
    for needles, cause in _CAUSES:
        if any(n in lowered for n in needles):
            return cause
    return Cause("unknown", "AWS's message doesn't name a known cause.", "Open a case with AWS Support citing the request ID.")


# --- the diagnosis ----------------------------------------------------------------


@dataclass
class Diagnosis:
    model_id: str
    region: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)
    causes: list[tuple[Cause, str]] = field(default_factory=list)  # (cause, evidence)
    credentials_ok: bool = True

    def add(self, title: str, lines: list[str]) -> None:
        self.sections.append((title, [sanitize(line) for line in lines]))

    def found(self, cause: Cause, evidence: str) -> None:
        """Record a cause; repeated evidence for the same cause is merged,
        a denied call's own message (CloudTrail) first."""
        evidence = sanitize(evidence)
        for i, (known, previous) in enumerate(self.causes):
            if known.category == cause.category:
                parts = [evidence, previous] if evidence.startswith("CloudTrail") else [previous, evidence]
                self.causes[i] = (known, "\n              ".join(parts))
                return
        self.causes.append((cause, evidence))


def _code(exc: Exception) -> str:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", type(exc).__name__)


def _base_model_id(model_id: str) -> str:
    head, _, rest = model_id.partition(".")
    return rest if head in {"us", "eu", "jp", "au", "apac", "global"} and rest else model_id


def role_arn_for(caller_arn: str, iam: Any) -> str | None:
    """arn:aws:sts::A:assumed-role/R/S -> the IAM role ARN (with its path,
    which SSO roles have and the simulator needs)."""
    m = re.match(r"arn:(aws[a-z-]*):sts::(\d{12}):assumed-role/([^/]+)/", caller_arn)
    if not m:
        return caller_arn if ":user/" in caller_arn or ":role/" in caller_arn else None
    partition, account, role = m.groups()
    try:
        return iam.get_role(RoleName=role)["Role"]["Arn"]
    except Exception:  # noqa: BLE001 -- no iam:GetRole: fall back to the well-known SSO path
        path = "aws-reserved/sso.amazonaws.com/" if role.startswith("AWSReservedSSO_") else ""
        return f"arn:{partition}:iam::{account}:role/{path}{role}"


def _simulate(iam: Any, role_arn: str, actions: list[str], resources: list[str],
              context: dict[str, str]) -> list[dict]:
    entries = [{"ContextKeyName": k, "ContextKeyValues": [v], "ContextKeyType": "string"} for k, v in context.items()]
    return iam.simulate_principal_policy(
        PolicySourceArn=role_arn, ActionNames=actions, ResourceArns=resources, ContextEntries=entries
    )["EvaluationResults"]


def _describe_result(r: dict) -> tuple[str, bool | None]:
    org = r.get("OrganizationsDecisionDetail", {}).get("AllowedByOrganizations")
    boundary = r.get("PermissionsBoundaryDecisionDetail", {}).get("AllowedByPermissionsBoundary")
    matched = ", ".join(sorted({s.get("SourcePolicyId", "?") for s in r.get("MatchedStatements", [])})) or "none"
    parts = [f"{r['EvalDecision']}", f"matched: {matched}"]
    if r.get("MissingContextValues"):
        parts.append(f"missing context: {', '.join(r['MissingContextValues'])}")
    if org is not None:
        parts.append(f"SCPs allow: {org}")
    if boundary is not None:
        parts.append(f"boundary allows: {boundary}")
    return "; ".join(parts), org


def diagnose(model_id: str, region: str, make: Callable[..., Any], *, role_arn: str | None = None,
             since_hours: int = 24, now: datetime | None = None) -> Diagnosis:
    d = Diagnosis(model_id=model_id, region=region)
    base_model = _base_model_id(model_id)

    # 1. caller
    try:
        caller_arn = make("sts", region_name=region).get_caller_identity()["Arn"]
    except Exception as exc:  # noqa: BLE001
        d.add("Caller", [f"STS rejected the credentials ({_code(exc)}) -- fix credentials first."])
        d.credentials_ok = False
        return d
    account = caller_arn.split(":")[4]
    role_arn = role_arn or role_arn_for(caller_arn, make("iam", region_name=region))
    d.add("Caller", [f"principal: {caller_arn}", f"IAM role simulated: {role_arn or 'unknown (pass --role-arn)'}"])

    bedrock = make("bedrock", region_name=region)

    # 2. model access, agreement, use-case form
    lines = []
    try:
        a = bedrock.get_foundation_model_availability(modelId=base_model)
        agreement = a.get("agreementAvailability", {})
        lines.append(
            f"authorization {a.get('authorizationStatus')}, entitlement {a.get('entitlementAvailability')}, "
            f"agreement {agreement.get('status')}, region {a.get('regionAvailability')}"
        )
        if agreement.get("errorMessage"):
            lines.append(f"agreement error: {agreement['errorMessage']}")
        if a.get("authorizationStatus") != "AUTHORIZED" or a.get("entitlementAvailability") != "AVAILABLE" \
                or agreement.get("status") != "AVAILABLE":
            d.found(_cause("model_access"), lines[0])
    except Exception as exc:  # noqa: BLE001
        lines.append(f"could not read model availability ({_code(exc)})")
    try:
        bedrock.get_use_case_for_model_access()
        lines.append("Anthropic use-case form: submitted")
    except Exception as exc:  # noqa: BLE001
        code = _code(exc)
        lines.append(f"Anthropic use-case form: {'NOT submitted' if code == 'ResourceNotFoundException' else 'unknown (' + code + ')'}")
        if code == "ResourceNotFoundException":
            d.found(_cause("use_case_form"), "GetUseCaseForModelAccess: not found")
    d.add("Model access", lines)

    # 3. profile routing
    profile_arn = f"arn:aws:bedrock:{region}:{account}:inference-profile/{model_id}"
    destinations: list[str] = []
    try:
        profile = bedrock.get_inference_profile(inferenceProfileIdentifier=model_id)
        profile_arn = profile.get("inferenceProfileArn", profile_arn)
        destinations = sorted({m["modelArn"].split(":")[3] for m in profile.get("models", []) if m.get("modelArn")})
        d.add("Inference profile", [f"{profile_arn} ({profile.get('status')})", f"routes to: {', '.join(destinations)}"])
    except Exception as exc:  # noqa: BLE001
        d.add("Inference profile", [f"could not read ({_code(exc)}); simulating the source region only"])
    destinations = destinations or [region]

    # 4. IAM simulation of the request as Bedrock evaluates it
    if role_arn:
        iam = make("iam", region_name=region)
        sim_lines = []
        checks = [("bedrock:InvokeModel", profile_arn, {"aws:RequestedRegion": region})] + [
            ("bedrock:InvokeModel", f"arn:aws:bedrock:{r}::foundation-model/{base_model}",
             {"bedrock:InferenceProfileArn": profile_arn, "aws:RequestedRegion": r})
            for r in destinations
        ] + [(a, "*", {}) for a in ("aws-marketplace:ViewSubscriptions", "aws-marketplace:Subscribe")]
        for action, resource, context in checks:
            try:
                (result,) = _simulate(iam, role_arn, [action], [resource], context)
            except Exception as exc:  # noqa: BLE001
                sim_lines.append(f"{action} on {resource}: could not simulate ({_code(exc)})")
                continue
            text, org_allowed = _describe_result(result)
            sim_lines.append(f"{action} on {resource}: {text}")
            if action.startswith("bedrock:"):
                if org_allowed is False:
                    d.found(_cause("scp"), f"simulation: SCPs deny {action} on {resource}")
                elif result["EvalDecision"] != "allowed":
                    d.found(_cause("identity_no_allow"), f"simulation: {result['EvalDecision']} on {resource}")
        sim_lines.append("(the simulator does not evaluate RCPs, session policies or VPC endpoint policies)")
        d.add("IAM simulation (with Bedrock's request context)", sim_lines)

    # 5. the denied calls themselves, from CloudTrail event history
    now = now or datetime.now(timezone.utc)
    trail_lines = []
    for r in sorted({region, *destinations}):
        try:
            events = make("cloudtrail", region_name=r).lookup_events(
                LookupAttributes=[{"AttributeKey": "EventSource", "AttributeValue": "bedrock.amazonaws.com"}],
                StartTime=now - timedelta(hours=since_hours), EndTime=now, MaxResults=50,
            )["Events"]
        except Exception as exc:  # noqa: BLE001
            trail_lines.append(f"{r}: could not read CloudTrail ({_code(exc)})")
            continue
        for e in events:
            detail = json.loads(e.get("CloudTrailEvent", "{}"))
            if detail.get("eventName") not in RUNTIME_EVENTS or not detail.get("errorCode"):
                continue
            message = detail.get("errorMessage", "")
            cause = classify_access_denied(message) if "AccessDenied" in detail["errorCode"] else None
            trail_lines.append(
                f"{detail.get('eventTime')} {r} {detail.get('eventName')} -> {detail['errorCode']} "
                f"(request {detail.get('requestID')}): {message}"
            )
            if cause:
                trail_lines.append(f"    => {cause.category}: {cause.explanation}")
                d.found(cause, f"CloudTrail {detail.get('requestID')}: {message}")
    if not any("->" in line for line in trail_lines):
        trail_lines.append(
            f"no failed Bedrock runtime calls in the last {since_hours}h (events can take ~15 min to appear)"
        )
    d.add("CloudTrail: failed Bedrock calls", trail_lines)
    return d


def _cause(category: str) -> Cause:
    return next(c for _, c in _CAUSES if c.category == category)


# --- reading one denial message -------------------------------------------------------

# The four questions a denial has to answer, in the order they are checked.
BUCKET_MARKETPLACE = "marketplace_subscription_permissions"
BUCKET_IAM = "iam_denial_on_inference_profile_or_foundation_model"
BUCKET_ACCOUNT = "model_access_or_account_restriction"
BUCKET_OTHER = "other_service_authorization"
BUCKET_UNRECOGNIZED = "unrecognized"

_IAM_STYLE = re.compile(
    r"not authorized to perform:?\s+(?P<action>[\w-]+:[\w*-]+)(?:\s+on resource:\s+(?P<resource>[^\s,]+))?"
)


@dataclass(frozen=True)
class DenialExplanation:
    bucket: str
    cause: Cause
    action: str | None  # the IAM action AWS says was denied, e.g. bedrock:InvokeModel
    resource: str | None  # the exact resource it was denied on (sanitized)
    resource_type: str | None  # inference-profile | foundation-model | other


def explain_denial(message: str) -> DenialExplanation:
    """Read an AccessDeniedException message: which action on which
    resource was denied, by which policy type or Bedrock gate, and which of
    the four buckets that falls in."""
    cause = classify_access_denied(message)
    match = _IAM_STYLE.search(message)
    action = match.group("action") if match else None
    resource = sanitize(match.group("resource")) if match and match.group("resource") else None
    resource_type = None
    if resource:
        resource_type = next((t for t in ("inference-profile", "foundation-model") if f":{t}/" in resource), "other")

    if cause.category == "marketplace" or (action or "").startswith("aws-marketplace:"):
        bucket, cause = BUCKET_MARKETPLACE, _cause("marketplace")
    elif action and not action.startswith("bedrock:"):
        bucket = BUCKET_OTHER  # e.g. kms:Decrypt, sts:..., another service's permission
    elif cause.category in {"scp", "rcp", "model_access", "model_not_offered", "use_case_form"}:
        bucket = BUCKET_ACCOUNT
    elif cause.category in {"identity_no_allow", "identity_explicit_deny", "permissions_boundary", "session_policy"}:
        bucket = BUCKET_IAM
    elif cause.category == "vpc_endpoint_policy":
        bucket = BUCKET_OTHER
    else:
        bucket = BUCKET_UNRECOGNIZED
    return DenialExplanation(bucket, cause, action, resource, resource_type)


def reproduce_converse(runtime: Any, model_id: str) -> dict[str, Any]:
    """Send ONE minimal Converse request (1-word prompt, maxTokens=1) and
    return what AWS said, sanitized. `runtime` must be a bedrock-runtime
    client configured for a single attempt, so a retryable error can't turn
    into several requests. A denied request is not billed; an allowed one
    costs a fraction of a cent."""
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        response = runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 1},
        )
    except ClientError as exc:
        error, meta = exc.response.get("Error", {}), exc.response.get("ResponseMetadata", {})
        headers = meta.get("HTTPHeaders", {})
        result: dict[str, Any] = {
            "outcome": "denied" if error.get("Code") == "AccessDeniedException" else "error",
            "code": error.get("Code"),
            "http_status": meta.get("HTTPStatusCode"),
            "request_id": meta.get("RequestId") or headers.get("x-amzn-requestid"),
            "error_type_header": headers.get("x-amzn-errortype"),
            "message": sanitize(error.get("Message", "")),
        }
        if result["outcome"] == "denied":
            result["explanation"] = explain_denial(error.get("Message", ""))
        elif error.get("Code") == "ValidationException":
            result["outcome"] = "authorized"  # the request got past authorization to validation
        return result
    except BotoCoreError as exc:
        return {"outcome": "not_sent", "code": type(exc).__name__, "message": sanitize(str(exc))}
    return {
        "outcome": "allowed",
        "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
        "stop_reason": response.get("stopReason"),
        "usage": response.get("usage", {}),
    }
