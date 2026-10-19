"""BedrockClaudeClient against stubbed boto3 clients (botocore Stubber):
request shape, response parsing, reasoning-block round trip, error
mapping, and how the agent, API and preflight behave when Bedrock fails.
No network calls -- every AWS response here is canned."""

from __future__ import annotations

import boto3
import pytest
from botocore.exceptions import NoCredentialsError, ReadTimeoutError
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from bankshield import config
from bankshield.api.app import app, get_llm_client
from bankshield.investigation import data_access
from bankshield.investigation.agent import run_investigation
from bankshield.investigation.bedrock_live import redact, run_preflight
from bankshield.investigation.case_store import get_case_store
from bankshield.investigation.llm_client import (
    BedrockClaudeClient,
    LLMServiceError,
    OpaqueBlock,
    TextBlock,
    ToolUseBlock,
    bedrock_settings,
)

MESSAGES = [{"role": "user", "content": [{"text": "hi"}]}]
TOOLS = [{"name": "get_risk_score", "description": "d", "input_schema": {"type": "object", "properties": {}}}]


def _stubbed(region: str = "us-east-1") -> tuple[BedrockClaudeClient, Stubber]:
    boto_client = boto3.client(
        "bedrock-runtime", region_name=region, aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return BedrockClaudeClient(region=region, boto_client=boto_client), Stubber(boto_client)


def _converse_response(content: list[dict], stop_reason: str = "end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 120, "outputTokens": 30, "totalTokens": 150},
        "metrics": {"latencyMs": 5},
    }


REASONING = {"reasoningContent": {"reasoningText": {"text": "thinking...", "signature": "sig-abc"}}}


# --- configuration -------------------------------------------------------------


def test_default_model_is_opus_5_5_us_inference_profile(monkeypatch):
    for var in (config.BEDROCK_MODEL_ID_ENV_VAR, config.BEDROCK_REGION_ENV_VAR, "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    assert bedrock_settings() == ("us.anthropic.claude-opus-5-5", "us-east-1")


def test_model_and_region_overridable_by_env(monkeypatch):
    monkeypatch.setenv(config.BEDROCK_MODEL_ID_ENV_VAR, "eu.anthropic.claude-opus-5-5")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.delenv(config.BEDROCK_REGION_ENV_VAR, raising=False)
    assert bedrock_settings() == ("eu.anthropic.claude-opus-5-5", "eu-west-1")
    monkeypatch.setenv(config.BEDROCK_REGION_ENV_VAR, "eu-central-1")
    assert bedrock_settings()[1] == "eu-central-1"


def test_boto_client_has_timeouts_and_bounded_adaptive_retries():
    client = BedrockClaudeClient(region="us-east-1")
    cfg = client._client.meta.config
    assert cfg.connect_timeout == config.BEDROCK_CONNECT_TIMEOUT_S
    assert cfg.read_timeout == config.BEDROCK_READ_TIMEOUT_S
    assert cfg.retries == {"total_max_attempts": config.BEDROCK_MAX_ATTEMPTS, "mode": "adaptive"}


# --- request / response --------------------------------------------------------


def test_converse_request_shape_has_no_sampling_params_or_forced_tool_choice():
    client, stub = _stubbed()
    stub.add_response(
        "converse",
        _converse_response([{"text": "ok"}]),
        {
            "modelId": client.model_id,
            "messages": MESSAGES,
            "system": [{"text": "sys"}],
            "inferenceConfig": {"maxTokens": config.BEDROCK_MAX_TOKENS},
            "toolConfig": {
                "tools": [{"toolSpec": {"name": "get_risk_score", "description": "d",
                                        "inputSchema": {"json": TOOLS[0]["input_schema"]}}}]
            },
        },
    )
    with stub:
        response = client.converse(MESSAGES, system="sys", tools=TOOLS)
    stub.assert_no_pending_responses()
    assert response.text() == "ok" and response.stop_reason == "end_turn"
    assert (response.input_tokens, response.output_tokens) == (120, 30)


def test_reasoning_and_tool_use_blocks_are_parsed_and_preserved():
    client, stub = _stubbed()
    stub.add_response(
        "converse",
        _converse_response(
            [REASONING, {"text": "Scoring."}, {"toolUse": {"toolUseId": "t1", "name": "get_risk_score", "input": {"transaction_id": "X"}}}],
            stop_reason="tool_use",
        ),
    )
    with stub:
        response = client.converse(MESSAGES, system="sys", tools=TOOLS)
    assert isinstance(response.content[0], OpaqueBlock) and response.content[0].raw == REASONING
    assert isinstance(response.content[1], TextBlock)
    assert response.tool_uses() == [ToolUseBlock(id="t1", name="get_risk_score", input={"transaction_id": "X"})]


# --- error mapping -------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "kind", "retryable"),
    [
        ("ThrottlingException", "throttled", True),
        ("ServiceQuotaExceededException", "throttled", True),
        ("ModelTimeoutException", "timeout", True),
        ("ServiceUnavailableException", "unavailable", True),
        ("InternalServerException", "unavailable", True),
        ("ModelErrorException", "model_error", False),
        ("ValidationException", "invalid_request", False),
        ("AccessDeniedException", "not_configured", False),
        ("ResourceNotFoundException", "not_configured", False),
        ("UnrecognizedClientException", "not_configured", False),
        ("SomethingNew", "model_error", False),
    ],
)
def test_client_errors_map_to_safe_llm_service_errors(code, kind, retryable):
    client, stub = _stubbed()
    secret_text = "User: arn:aws:iam::123456789012:user/alice is not authorized"
    stub.add_client_error("converse", service_error_code=code, service_message=secret_text, http_status_code=400)
    with stub, pytest.raises(LLMServiceError) as excinfo:
        client.converse(MESSAGES, system="sys", tools=[])
    err = excinfo.value
    assert (err.kind, err.retryable, err.aws_code) == (kind, retryable, code)
    assert "123456789012" not in str(err) and "arn:" not in str(err)
    # AWS's own explanation is kept for local diagnostics, sanitized
    assert "is not authorized" in err.aws_message and "123456789012" not in err.aws_message
    assert err.__cause__ is None  # raw AWS error not chained into tracebacks


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"), "timeout"),
        (NoCredentialsError(), "not_configured"),
    ],
)
def test_transport_and_credential_errors_map(monkeypatch, exc, kind):
    client, _ = _stubbed()

    def boom(**_):
        raise exc

    monkeypatch.setattr(client._client, "converse", boom)
    with pytest.raises(LLMServiceError) as excinfo:
        client.converse(MESSAGES, system="sys", tools=[])
    assert excinfo.value.kind == kind


# --- agent / API behavior on Bedrock responses and failures ----------------------


def _tier1_txn() -> str:
    return next(t["transaction_id"] for t in data_access.sample_transactions(n=10) if t["risk_tier"] == "tier_1")


def test_agent_round_trips_reasoning_blocks_on_the_next_turn():
    """Opus 5.5's signed thinking must be sent back verbatim with the tool
    results, or Bedrock rejects the conversation."""
    client, stub = _stubbed()
    txn = _tier1_txn()
    stub.add_response(
        "converse",
        _converse_response(
            [REASONING, {"toolUse": {"toolUseId": "t1", "name": "get_risk_score", "input": {"transaction_id": txn}}}],
            stop_reason="tool_use",
        ),
    )
    stub.add_response("converse", _converse_response([{"text": "Done. inconclusive_monitor"}]))
    seen: list[list[dict]] = []
    real = client._client.converse

    def spy(**kwargs):
        seen.append([dict(m) for m in kwargs["messages"]])
        return real(**kwargs)

    with stub:
        client._client.converse = spy
        result = run_investigation(txn, client)
    assert seen[1][1] == {"role": "assistant", "content": [REASONING, seen[1][1]["content"][1]]}
    assert result.tool_calls[0].tool_name == "get_risk_score" and not result.tool_calls[0].is_error


@pytest.mark.parametrize("stop_reason", ["guardrail_intervened", "content_filtered", "malformed_tool_use"])
def test_abnormal_stop_runs_no_tools_and_is_flagged(stop_reason):
    client, stub = _stubbed()
    txn = _tier1_txn()
    stub.add_response(
        "converse",
        _converse_response(
            [{"toolUse": {"toolUseId": "t1", "name": "create_case",
                          "input": {"transaction_id": txn, "disposition": "confirmed_fraud", "summary": "x"}}}],
            stop_reason=stop_reason,
        ),
    )
    with stub:
        result = run_investigation(txn, client)
    assert result.tool_calls == [] and result.pending_approvals == []
    assert f"stopped by the model: {stop_reason}" in result.narrative


def test_failure_after_case_proposal_keeps_it_pending_and_creates_no_case():
    client, stub = _stubbed()
    txn = _tier1_txn()
    stub.add_response(
        "converse",
        _converse_response(
            [{"toolUse": {"toolUseId": "t1", "name": "create_case",
                          "input": {"transaction_id": txn, "disposition": "confirmed_fraud", "summary": "x"}}}],
            stop_reason="tool_use",
        ),
    )
    stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)
    cases_before = len(get_case_store().list_all())
    with stub, pytest.raises(LLMServiceError) as excinfo:
        run_investigation(txn, client)
    assert len(excinfo.value.pending_approval_ids) == 1
    assert len(get_case_store().list_all()) == cases_before


class _FailingClient:
    model_id = "us.anthropic.claude-opus-5-5"

    def __init__(self, error: LLMServiceError):
        self.error = error

    def converse(self, messages, system, tools):
        raise self.error


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMServiceError("throttled", "The model is rate-limited; retry shortly.", retryable=True), 503),
        (LLMServiceError("not_configured", "Access to the configured model is denied.", retryable=False,
                         aws_message="User: arn:aws:iam::****9012:user/x is not authorized ..."), 502),
    ],
)
def test_api_maps_bedrock_failures_to_safe_5xx_and_saves_nothing(auth_headers, error, status):
    txn = _tier1_txn()
    app.dependency_overrides[get_llm_client] = lambda: _FailingClient(error)
    try:
        with TestClient(app) as http:
            saved_before = http.get(f"/investigations/{txn}", headers=auth_headers("analyst")).json()
            response = http.post("/investigations", json={"transaction_id": txn}, headers=auth_headers("analyst"))
            assert response.status_code == status
            body = response.json()["detail"]
            assert "not authorized" not in response.text  # AWS's message never reaches API callers
            assert body["error"] == error.kind and body["retryable"] is error.retryable
            assert ("Retry-After" in response.headers) is error.retryable
            # nothing saved for the failed run (an earlier test's result, if any, is untouched)
            assert http.get(f"/investigations/{txn}", headers=auth_headers("analyst")).json() == saved_before
    finally:
        app.dependency_overrides.pop(get_llm_client, None)


# --- preflight and redaction ------------------------------------------------------


def _preflight_factory(stubs: dict[str, Stubber]):
    clients = {}

    def factory(service, region_name):
        if service not in clients:
            clients[service] = boto3.client(
                service, region_name=region_name, aws_access_key_id="testing", aws_secret_access_key="testing"
            )
            stubs[service] = Stubber(clients[service])
            stubs[service].activate()
        return clients[service]

    return factory


def test_preflight_reports_invalid_credentials_as_blocker(monkeypatch):
    monkeypatch.delenv(config.BEDROCK_MODEL_ID_ENV_VAR, raising=False)
    stubs: dict[str, Stubber] = {}
    factory = _preflight_factory(stubs)
    factory("sts", "us-east-1")
    stubs["sts"].add_client_error("get_caller_identity", service_error_code="InvalidClientTokenId", http_status_code=403)
    report = run_preflight(client_factory=factory)
    assert not report.ready and "InvalidClientTokenId" in report.blockers[0]


def test_preflight_ready_and_reports_missing_model_access(monkeypatch):
    monkeypatch.delenv(config.BEDROCK_MODEL_ID_ENV_VAR, raising=False)
    stubs: dict[str, Stubber] = {}
    factory = _preflight_factory(stubs)
    factory("sts", "us-east-1"), factory("bedrock", "us-east-1")
    profile_arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5-5"

    def queue(entitlement):
        stubs["sts"].add_response(
            "get_caller_identity",
            {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/r/s", "UserId": "AROA:s"},
        )
        stubs["bedrock"].add_response(
            "get_inference_profile",
            {"inferenceProfileName": "US Claude Opus 5.5", "inferenceProfileArn": profile_arn,
             "inferenceProfileId": "us.anthropic.claude-opus-5-5", "status": "ACTIVE", "type": "SYSTEM_DEFINED",
             "models": [{"modelArn": f"arn:aws:bedrock:{r}::foundation-model/anthropic.claude-opus-5-5"}
                        for r in ("us-east-1", "us-east-2", "us-west-2")]},
            {"inferenceProfileIdentifier": "us.anthropic.claude-opus-5-5"},
        )
        stubs["bedrock"].add_response(
            "get_foundation_model_availability",
            {"modelId": "anthropic.claude-opus-5-5", "agreementAvailability": {"status": "AVAILABLE"},
             "authorizationStatus": "AUTHORIZED", "entitlementAvailability": entitlement,
             "regionAvailability": "AVAILABLE"},
            {"modelId": "anthropic.claude-opus-5-5"},
        )

    queue("AVAILABLE")
    report = run_preflight(client_factory=factory)
    assert report.ready, report.blockers
    assert any("us-east-2" in line for line in report.ok)
    assert not any("123456789012" in line for line in report.ok)  # account masked

    queue("NOT_AVAILABLE")
    report = run_preflight(client_factory=factory)
    assert not report.ready and "enable model access" in report.blockers[0]


def test_redact_removes_aws_identifiers():
    text = {
        "a": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5-5",
        "b": ["key AKIAABCDEFGHIJKLMNOP and account 123456789012", "req 1b4e28ba-2fa1-11d2-883f-0016d3cca427"],
        "c": 42,
    }
    out = redact(text)
    flat = str(out)
    assert "123456789012" not in flat and "AKIA" not in flat and "1b4e28ba" not in flat and "arn:" not in flat
    assert out["c"] == 42
