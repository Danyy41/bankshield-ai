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
(maxTokens=1, a 1-word prompt) and prints the live error. If it is denied
it costs nothing; if it is allowed it costs well under $0.001.

Usage:
    python scripts/bedrock_access_diagnose.py [--hours 24] [--role-arn ARN]
    python scripts/bedrock_access_diagnose.py --reproduce --confirm-billable
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boto3  # noqa: E402

from bankshield.investigation.bedrock_diagnostics import classify_access_denied, diagnose, sanitize  # noqa: E402
from bankshield.investigation.llm_client import bedrock_settings  # noqa: E402


def reproduce(model_id: str, region: str) -> None:
    from botocore.exceptions import ClientError

    client = boto3.client("bedrock-runtime", region_name=region)
    print("\n== Live reproduction (one Converse call, maxTokens=1) ==")
    try:
        r = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 1},
        )
        print(f"  ALLOWED: stopReason={r['stopReason']}, request {r['ResponseMetadata']['RequestId']} "
              "-- access works now (this call was billed a fraction of a cent)")
    except ClientError as exc:
        err, meta = exc.response.get("Error", {}), exc.response.get("ResponseMetadata", {})
        print(f"  code:       {err.get('Code')}")
        print(f"  HTTP:       {meta.get('HTTPStatusCode')}")
        print(f"  request ID: {meta.get('RequestId')}")
        print(f"  message:    {sanitize(err.get('Message', ''))}")
        if err.get("Code") == "AccessDeniedException":
            cause = classify_access_denied(err.get("Message", ""))
            print(f"  => {cause.category}: {cause.explanation}\n     fix: {cause.fix}")
        elif err.get("Code") == "ValidationException":
            print("  => authorization PASSED (the request reached validation); access is not the problem")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=int, default=24, help="CloudTrail look-back window")
    parser.add_argument("--role-arn", help="IAM role to simulate (default: derived from the caller)")
    parser.add_argument("--reproduce", action="store_true", help="also send one tiny Converse call")
    parser.add_argument("--confirm-billable", action="store_true", help="required with --reproduce")
    args = parser.parse_args()
    if args.reproduce and not args.confirm_billable:
        parser.error("--reproduce sends a (possibly billable) Converse call; add --confirm-billable")

    model_id, region = bedrock_settings()
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
