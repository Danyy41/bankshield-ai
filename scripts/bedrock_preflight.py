"""Check that this machine can run BankShield investigations on live
Amazon Bedrock -- using only free API calls, never the model itself.

Checks, in order: AWS credentials resolve through the standard provider
chain and are valid (sts:GetCallerIdentity), the configured inference
profile exists in the configured region (bedrock:GetInferenceProfile), and
the account has model access to Claude (bedrock:GetFoundationModelAvailability).
Prints exactly what is missing. Exit code 0 = ready, 1 = blocked.

Configuration (all optional):
    BANKSHIELD_BEDROCK_MODEL_ID   default us.anthropic.claude-sonnet-5-5
    BANKSHIELD_BEDROCK_REGION     else AWS_REGION / AWS_DEFAULT_REGION, else us-east-1
    AWS_PROFILE / AWS_* vars      standard boto3 credential sources

Usage:
    python scripts/bedrock_preflight.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankshield.investigation.bedrock_live import run_preflight  # noqa: E402


def main() -> int:
    report = run_preflight()
    print(f"Bedrock preflight: model {report.model_id} in {report.region} (no inference calls made)\n")
    for line in report.ok:
        print(f"[ OK ] {line}")
    for line in report.warnings:
        print(f"[WARN] {line}")
    for line in report.blockers:
        print(f"[FAIL] {line}")
    print("\nReady for a live investigation." if report.ready else "\nNot ready: fix the [FAIL] items above.")
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
