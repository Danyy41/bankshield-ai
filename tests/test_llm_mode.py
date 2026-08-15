"""Tests for the BANKSHIELD_LLM_MODE deployment-mode switch (api/app.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bankshield import config
from bankshield.api.app import app, get_llm_client
from bankshield.investigation import data_access
from bankshield.investigation.llm_client import AutoFakeLLMClient, BedrockClaudeClient


def test_default_mode_is_offline_with_no_env_var(monkeypatch):
    monkeypatch.delenv(config.LLM_MODE_ENV_VAR, raising=False)
    client = get_llm_client()
    assert isinstance(client, AutoFakeLLMClient)


def test_offline_mode_explicit(monkeypatch):
    monkeypatch.setenv(config.LLM_MODE_ENV_VAR, "offline")
    client = get_llm_client()
    assert isinstance(client, AutoFakeLLMClient)


def test_bedrock_mode_constructs_bedrock_client_without_network_call(monkeypatch):
    monkeypatch.setenv(config.LLM_MODE_ENV_VAR, "bedrock")
    client = get_llm_client()
    assert isinstance(client, BedrockClaudeClient)


def test_invalid_mode_raises_clear_error(monkeypatch):
    monkeypatch.setenv(config.LLM_MODE_ENV_VAR, "not-a-real-mode")
    with pytest.raises(RuntimeError, match="not-a-real-mode"):
        get_llm_client()


def test_api_works_with_zero_configuration(monkeypatch):
    """The public-demo default: no BANKSHIELD_LLM_MODE set, no
    dependency_overrides -- the app must still serve a full investigation
    end to end, since offline is the default mode."""
    monkeypatch.delenv(config.LLM_MODE_ENV_VAR, raising=False)
    assert app.dependency_overrides.get(get_llm_client) is None

    store = data_access.get_store()
    candidates = store.transactions[
        (store.transactions["is_fraud"] == 1) & (store.transactions["graph_suspicious_neighbor_count"] > 0)
    ]
    txn_id = None
    for _, row in candidates.iterrows():
        risk = data_access.get_risk_score(row["transaction_id"])
        if risk["risk_tier"] == "tier_1":
            txn_id = row["transaction_id"]
            break
    if txn_id is None:
        pytest.skip("no tier_1 transaction available")

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/investigations", json={"transaction_id": txn_id})
        assert response.status_code == 200
        assert response.json()["disposition"] == "confirmed_fraud"
