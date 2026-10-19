"""Find which Claude model this AWS account can actually invoke on Bedrock.

Free by default: for each candidate inference profile it reads the profile
and the model's availability (read-only control-plane calls, no inference)
and prints a table. These checks can pass for a model the account still
can't invoke ("<model> is not available for this account"), so the only
definitive test is a real request:

--confirm-billable sends, for each candidate in order, ONE Converse
request (1-word prompt, maxTokens=1, a single attempt with no retries) and
stops at the first model that answers. Denied requests are not billed, so
at most one request is billed -- a few input tokens and 1 output token,
well under $0.001. For each candidate it prints the request ID and AWS's
sanitized reason.

The identity needs bedrock:InvokeModel on every candidate it should test
(scripts/render_bedrock_iam_policy.py --model-id ... for each); a candidate
without it is reported as an IAM denial, which says nothing about whether
the account could use that model.

Usage:
    python scripts/bedrock_model_access_test.py                    # free checks only
    python scripts/bedrock_model_access_test.py --confirm-billable # + one tiny request per candidate
    python scripts/bedrock_model_access_test.py --model-id us.anthropic.claude-sonnet-5-5 --confirm-billable
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402

from bankshield.investigation.bedrock_diagnostics import reproduce_converse  # noqa: E402
from bankshield.investigation.bedrock_live import _base_model_id  # noqa: E402
from bankshield.investigation.llm_client import bedrock_settings  # noqa: E402

# Newest first; the ARN-versioned Sonnet 4.5 profile is the long-standing fallback.
DEFAULT_CANDIDATES = [
    "us.anthropic.claude-sonnet-5-5",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
]


def _code(exc: Exception) -> str:
    return (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", type(exc).__name__)


def free_checks(bedrock, model_id: str) -> str:
    parts = []
    try:
        profile = bedrock.get_inference_profile(inferenceProfileIdentifier=model_id)
        regions = sorted({m["modelArn"].split(":")[3] for m in profile.get("models", []) if m.get("modelArn")})
        parts.append(f"profile {profile.get('status')} -> {', '.join(regions)}")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"profile: {_code(exc)}")
    try:
        a = bedrock.get_foundation_model_availability(modelId=_base_model_id(model_id))
        parts.append(
            f"availability: auth {a.get('authorizationStatus')}, entitlement {a.get('entitlementAvailability')}, "
            f"agreement {a.get('agreementAvailability', {}).get('status')}, region {a.get('regionAvailability')}"
        )
    except Exception as exc:  # noqa: BLE001
        parts.append(f"availability: {_code(exc)}")
    return "; ".join(parts)


def main() -> int:
    configured, region = bedrock_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", action="append", help="candidate inference profile (repeatable)")
    parser.add_argument("--confirm-billable", action="store_true",
                        help="send one tiny Converse request per candidate until one is allowed")
    args = parser.parse_args()
    candidates = args.model_id or list(dict.fromkeys([configured, *DEFAULT_CANDIDATES]))

    print(f"Model access test in {region} -- candidates: {', '.join(candidates)}\n")
    bedrock = boto3.client("bedrock", region_name=region)
    for model_id in candidates:
        print(f"[free] {model_id}: {free_checks(bedrock, model_id)}")
    if not args.confirm_billable:
        print("\nNo request sent. These checks can't prove invocation works; re-run with --confirm-billable.")
        return 0

    runtime = boto3.client(
        "bedrock-runtime", region_name=region,
        config=Config(connect_timeout=10, read_timeout=60, retries={"total_max_attempts": 1, "mode": "standard"}),
    )
    print()
    for model_id in candidates:
        r = reproduce_converse(runtime, model_id)
        if r["outcome"] == "allowed":
            print(f"[ OK ] {model_id}: INVOCABLE (request {r['request_id']}, usage {r['usage']})")
            print(f"\nUse it:  BANKSHIELD_BEDROCK_MODEL_ID={model_id}")
            return 0
        reason = r.get("message", "")
        e = r.get("explanation")
        label = f"{e.bucket} / {e.cause.category}" if e else r.get("outcome")
        print(f"[DENY] {model_id}: {r.get('code')} (request {r.get('request_id')}) [{label}]\n       {reason}")
    print("\nNo candidate is invocable from this account; see the reasons above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
