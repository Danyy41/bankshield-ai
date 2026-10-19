# Optional integration: Claude on Amazon Bedrock

> **Status: optional and unverified.** BankShield runs offline by default, and
> every result in this repository comes from the deterministic offline client.
> The Bedrock integration is covered by tests that stub AWS, but **no live
> Bedrock inference has succeeded for this project**, so its end-to-end
> behavior against a real model is unverified.

## What it is

`BedrockClaudeClient` (`src/bankshield/investigation/llm_client.py`) is a
second implementation of the same `LLMClient` interface the offline client
uses. It speaks the Bedrock **Converse** API, so the agent loop, tools,
guardrails, citation filtering and human-approval gate are identical in both
modes.

| Setting | Default | Notes |
|---|---|---|
| `BANKSHIELD_LLM_MODE` | `offline` | Set to `bedrock` to opt in |
| `BANKSHIELD_BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-5-5` | Any Claude cross-region inference profile, e.g. `us.anthropic.claude-opus-5-5` |
| `BANKSHIELD_BEDROCK_REGION` | `AWS_REGION`, else `us-east-1` | Region the profile is called in |

The US geographic profile keeps traffic inside US regions, which suits a
bank's data-residency posture better than `global.` profiles.

## Security and failure handling

- **Credentials:** only boto3's standard provider chain (IAM Identity Center
  profile, ECS task role, EC2 instance profile, or short-lived environment
  credentials). The app never accepts, stores or logs a key.
- **Least-privilege IAM:** [`deploy/iam/bedrock-invoke-policy.json`](../deploy/iam/bedrock-invoke-policy.json)
  grants `bedrock:InvokeModel` (which authorizes Converse) on the inference
  profile, and on its foundation model only when reached through that
  profile (`bedrock:InferenceProfileArn` condition). It has no wildcards and
  no Marketplace actions. Generate it for any model with
  `python scripts/render_bedrock_iam_policy.py --account-id <id> --model-id <profile>`;
  a test checks the committed policies against the generator.
- **Fail closed:** connect and read timeouts, bounded adaptive retries, then
  a caller-safe `LLMServiceError`. The API returns `503` (with
  `Retry-After`) or `502`, and saves nothing. A case already proposed stays
  a *pending* approval, so a reviewer still decides. AWS's own error text is
  logged only in sanitized form (account IDs masked) and never returned to
  API or A2A callers.
- Tools run only on a `tool_use` stop. Signed reasoning blocks are passed back
  unchanged between turns. No sampling parameters or forced `tool_choice` are
  sent.

## Tools

| Script | Cost | Purpose |
|---|---|---|
| `scripts/bedrock_preflight.py` | Free | Checks credentials, the inference profile and model access using read-only calls, and says what's missing |
| `scripts/bedrock_access_diagnose.py` | Free by default | Explains why a call is denied, using read-only checks (CloudTrail, model access, an IAM simulation with Bedrock's request context). `--reproduce-only --confirm-billable` sends one `maxTokens=1` request |
| `scripts/run_live_bedrock_investigation.py` | Billable, needs `--confirm-billable` | Runs one investigation, verifies tool use, citations and the approval gate, and saves a redacted transcript only if every check passes |
| `scripts/13_evaluate_investigations.py --live --confirm-billable` | Billable | Runs the Phase 4 evaluation against the live model |
