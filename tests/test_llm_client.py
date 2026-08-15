"""Tests for the LLM client abstraction (llm_client.py): the fake/offline
implementations exercised by the rest of the suite, plus a construction-only
smoke test for the real Bedrock client (no network call, since this
environment has no AWS credentials)."""

from __future__ import annotations

import pytest

from bankshield import config
from bankshield.investigation import data_access
from bankshield.investigation.llm_client import (
    AutoFakeLLMClient,
    BedrockClaudeClient,
    FakeLLMClient,
    LLMResponse,
    TextBlock,
    default_investigation_script,
)


def _end_turn_step(messages):
    return LLMResponse(
        stop_reason="end_turn",
        content=[TextBlock(text="done")],
        input_tokens=1,
        output_tokens=1,
        model_id="fake-claude",
    )


def test_fake_llm_client_requires_at_least_one_step():
    with pytest.raises(ValueError):
        FakeLLMClient([])


def test_fake_llm_client_raises_when_script_exhausted():
    client = FakeLLMClient([_end_turn_step])
    messages = [{"role": "user", "content": [{"text": "hi"}]}]
    client.converse(messages, system="sys", tools=[])  # consumes the one step

    # A second call with a conversation that now has two assistant turns
    # asks for a step index the script doesn't have.
    messages_with_two_assistant_turns = [
        {"role": "user", "content": []},
        {"role": "assistant", "content": []},
        {"role": "user", "content": []},
        {"role": "assistant", "content": []},
    ]
    with pytest.raises(RuntimeError):
        client.converse(messages_with_two_assistant_turns, system="sys", tools=[])


def test_fake_llm_client_turn_index_derived_from_messages_not_call_count():
    """Two independent conversations against the same client instance must
    each start at step 0 -- turn index comes from the conversation's own
    assistant-turn count, not a mutable counter on the client."""
    client = FakeLLMClient([_end_turn_step])

    conversation_a = [{"role": "user", "content": [{"text": "hi"}]}]
    conversation_b = [{"role": "user", "content": [{"text": "hi"}]}]

    response_a = client.converse(conversation_a, system="sys", tools=[])
    response_b = client.converse(conversation_b, system="sys", tools=[])

    assert response_a.stop_reason == "end_turn"
    assert response_b.stop_reason == "end_turn"


def test_auto_fake_llm_client_resolves_transaction_id_from_briefing():
    store = data_access.get_store()
    txn_id = store.transactions.iloc[0]["transaction_id"]
    client = AutoFakeLLMClient()

    briefing = f'{{"transaction_id": "{txn_id}", "risk_score": 0.1}}'
    messages = [{"role": "user", "content": [{"text": briefing}]}]
    response = client.converse(messages, system="sys", tools=[])

    assert response.stop_reason == "tool_use"
    assert response.tool_uses()[0].input["transaction_id"] == txn_id


def test_auto_fake_llm_client_serves_multiple_transactions_from_one_instance():
    store = data_access.get_store()
    txn_a, txn_b = store.transactions.iloc[0]["transaction_id"], store.transactions.iloc[1]["transaction_id"]
    client = AutoFakeLLMClient()

    for txn_id in (txn_a, txn_b):
        messages = [{"role": "user", "content": [{"text": f'{{"transaction_id": "{txn_id}"}}'}]}]
        response = client.converse(messages, system="sys", tools=[])
        assert response.tool_uses()[0].input["transaction_id"] == txn_id


def test_default_investigation_script_has_multiple_steps():
    store = data_access.get_store()
    txn_id = store.transactions.iloc[0]["transaction_id"]
    steps = default_investigation_script(txn_id)
    assert len(steps) >= 3


def test_bedrock_claude_client_constructs_without_network_call():
    client = BedrockClaudeClient()
    assert client.model_id == config.BEDROCK_MODEL_ID_DEFAULT
    assert client.region == config.BEDROCK_REGION_DEFAULT
