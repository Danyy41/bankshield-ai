"""Print the least-privilege IAM policy BankShield needs for one or more
Bedrock inference profiles. Makes no AWS calls.

Usage:
    python scripts/render_bedrock_iam_policy.py --account-id 123456789012
    python scripts/render_bedrock_iam_policy.py --account-id 123456789012 \\
        --model-id us.anthropic.claude-sonnet-5-5 --model-id us.anthropic.claude-opus-5-5

Defaults to the configured model (BANKSHIELD_BEDROCK_MODEL_ID, else the
config default) and region. Paste the output into the identity's inline
policy, replacing the previous BankShield statements.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankshield.investigation.bedrock_live import invoke_policy  # noqa: E402
from bankshield.investigation.llm_client import bedrock_settings  # noqa: E402


def main() -> None:
    model_id, region = bedrock_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--account-id", default="ACCOUNT_ID")
    parser.add_argument("--model-id", action="append", help="inference profile ID (repeatable)")
    parser.add_argument("--region", default=region, help="region the inference profile is called in")
    args = parser.parse_args()
    print(json.dumps(invoke_policy(args.model_id or [model_id], args.account_id, args.region), indent=2))


if __name__ == "__main__":
    main()
