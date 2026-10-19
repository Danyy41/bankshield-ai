"""Find out why Amazon Bedrock denies BankShield's Converse call.

Free by default: only read-only AWS APIs (STS, IAM GetRole and
SimulatePrincipalPolicy, Bedrock model-availability/use-case/inference-
profile reads, CloudTrail LookupEvents). It never weakens or changes any
policy. Prints, sanitized for local display (account IDs masked to their
last four digits, SSO session names hidden): the denied calls' error code,
AWS's message, request ID and region from CloudTrail; model access and
agreement state; and an IAM simulation of the exact request context
(profile in the source region, foundation model in each destination region
with bedrock:InferenceProfileArn) including SCP and boundary verdicts.

--reproduce --confirm-billable additionally sends ONE Converse request
(maxTokens=1, a 1-word prompt, a single attempt with no retries) and prints
AWS's own error: code, HTTP status, request ID, the sanitized message, and
which action on which resource was denied by which policy type -- i.e.
whether it is a Marketplace-subscription permission, an IAM denial on the
inference profile or foundation model, a model-access/account restriction,
or another service's authorization. --reproduce-only skips the read-only
checks (useful for a restricted user that can't read IAM/CloudTrail). A
denied request is not billed; an allowed one costs well under $0.001.

Usage:
    python scripts/bedrock_access_diagnose.py [--hours 24] [--role-arn ARN]
    python scripts/bedrock_access_diagnose.py --reproduce-only --confirm-billable
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boto3  # noqa: E402

from bankshield.investigation.bedrock_diagnostics import (  # noqa: E402
    BUCKET_ACCOUNT,
    BUCKET_IAM,
    BUCKET_MARKETPLACE,
    BUCKET_OTHER,
    BUCKET_UNRECOGNIZED,
    diagnose,
    reproduce_converse,
)
from bankshield.investigation.llm_client import bedrock_settings  # noqa: E402


BUCKET_LABELS = {
    BUCKET_MARKETPLACE: "Missing AWS Marketplace subscription permissions",
    BUCKET_IAM: "IAM denial on the inference profile or foundation model",
    BUCKET_ACCOUNT: "Model access or account-level restriction (SCP/RCP/model access/use-case form)",
    BUCKET_OTHER: "Another service's authorization requirement",
    BUCKET_UNRECOGNIZED: "Not recognized -- read the message; cite the request ID to AWS Support",
}


def reproduce(model_id: str, region: str) -> None:
    from botocore.config import Config

    # One attempt only: no retries, so a throttle can't become several requests.
    runtime = boto3.client(
        "bedrock-runtime", region_name=region,
        config=Config(connect_timeout=10, read_timeout=60, retries={"total_max_attempts": 1, "mode": "standard"}),
    )
    print(f"\n== Controlled Converse request: {model_id} in {region} (1 attempt, maxTokens=1) ==")
    r = reproduce_converse(runtime, model_id)
    if r["outcome"] == "allowed":
        print(f"  ALLOWED: request {r['request_id']}, stopReason {r['stop_reason']}, usage {r['usage']}")
        print("  Access works now. (This request was billed: a fraction of a cent.)")
        return
    if r["outcome"] == "not_sent":
        print(f"  not sent ({r['code']}): {r['message']}")
        return
    print(f"  code:            {r['code']}")
    print(f"  HTTP status:     {r['http_status']}")
    print(f"  request ID:      {r['request_id']}")
    print(f"  x-amzn-ErrorType:{' ' + str(r['error_type_header'])}")
    print(f"  AWS message:     {r['message']}")
    if r["outcome"] == "authorized":
        print("  => Authorization PASSED (the request reached validation). Access is not the problem.")
        return
    e = r.get("explanation")
    if e is None:
        return
    print(f"\n  denied action:   {e.action or '(not stated in the message)'}")
    print(f"  denied resource: {e.resource or '(not stated)'}" + (f"  [{e.resource_type}]" if e.resource_type else ""))
    print(f"  decided by:      {e.cause.category} -- {e.cause.explanation}")
    print(f"  => {BUCKET_LABELS[e.bucket]}")
    print(f"     fix: {e.cause.fix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=int, default=24, help="CloudTrail look-back window")
    parser.add_argument("--role-arn", help="IAM role to simulate (default: derived from the caller)")
    parser.add_argument("--reproduce", action="store_true", help="also send one tiny Converse call")
    parser.add_argument("--reproduce-only", action="store_true",
                        help="send only the one tiny Converse call; skip the read-only checks")
    parser.add_argument("--confirm-billable", action="store_true", help="required with --reproduce(-only)")
    args = parser.parse_args()
    if (args.reproduce or args.reproduce_only) and not args.confirm_billable:
        parser.error("--reproduce sends a Converse call (billed only if allowed); add --confirm-billable")

    model_id, region = bedrock_settings()
    if args.reproduce_only:
        reproduce(model_id, region)
        return 0
    print(f"Bedrock access diagnosis: {model_id} from {region} (read-only, no inference)")
    d = diagnose(model_id, region, boto3.client, role_arn=args.role_arn, since_hours=args.hours)
    for title, lines in d.sections:
        print(f"\n== {title} ==")
        for line in lines:
            print(f"  {line}")

    print("\n== Verdict ==")
    if not d.credentials_ok:
        print("  Credentials are not valid, so nothing else could be checked.")
        return 2
    if not d.causes:
        print("  No cause identified from free checks. Make the failing call again (or use --reproduce),")
        print("  wait ~15 minutes for CloudTrail, and re-run; RCPs, session policies and VPC endpoint")
        print("  policies are only visible in the denied call's message.")
    # A denied call's own message (CloudTrail) outranks inferences from the simulator.
    for cause, evidence in sorted(d.causes, key=lambda ce: not ce[1].startswith("CloudTrail")):
        print(f"  [{cause.category}] {cause.explanation}\n    evidence: {evidence}\n    fix: {cause.fix}")
    if args.reproduce:
        reproduce(model_id, region)
    return 0 if not d.causes else 1


if __name__ == "__main__":
    sys.exit(main())
