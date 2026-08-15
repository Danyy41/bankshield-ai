"""LLM client abstraction, with a Bedrock/Claude implementation and a fake
implementation for offline tests and the evaluation harness.

Message and tool-spec shapes here follow Amazon Bedrock's `converse` API
directly (https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
rather than introducing a separate internal message format that would need
translating both ways -- `agent.py` builds Converse-shaped messages once and
both `BedrockClaudeClient` and `FakeLLMClient` consume the same shape, so
swapping implementations (e.g. in tests) requires no adapter.

Why an interface at all, given there's one real backend: this sandbox has
no AWS credentials, so the test suite and evaluation harness must run
without ever calling Bedrock. `FakeLLMClient` is not a mock of the SDK --
it's a legitimate, deterministic implementation of the same contract,
letting `agent.py`'s tool-use loop be exercised end-to-end in CI.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from bankshield import config


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


ContentBlock = TextBlock | ToolUseBlock


@dataclass
class LLMResponse:
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"
    content: list[ContentBlock]
    input_tokens: int
    output_tokens: int
    model_id: str

    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class LLMClient(ABC):
    """A single request/response turn against a Claude-family model,
    Bedrock Converse-shaped in and out. The caller (agent.py) owns the
    tool-use loop; implementations only need to answer one turn at a time.
    """

    model_id: str

    @abstractmethod
    def converse(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


def _bedrock_tool_config(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": {"json": tool["input_schema"]},
                }
            }
            for tool in tools
        ]
    }


class BedrockClaudeClient(LLMClient):
    """Real Amazon Bedrock backend for Claude, via the Converse API.

    Requires `boto3` and AWS credentials with `bedrock:InvokeModel`
    permission for the configured model. Not exercised by this repo's test
    suite (no AWS credentials in CI) -- see FakeLLMClient for the path the
    tests and evaluation harness actually run.
    """

    def __init__(self, model_id: str | None = None, region: str | None = None):
        import boto3  # local import: keep boto3 optional for anyone only running the offline path

        self.model_id = model_id or config.BEDROCK_MODEL_ID_DEFAULT
        self.region = region or config.BEDROCK_REGION_DEFAULT
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def converse(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "system": [{"text": system}],
            "inferenceConfig": {"maxTokens": 4096},
        }
        if tools:
            kwargs["toolConfig"] = _bedrock_tool_config(tools)

        response = self._client.converse(**kwargs)

        content: list[ContentBlock] = []
        for block in response["output"]["message"]["content"]:
            if "text" in block:
                content.append(TextBlock(text=block["text"]))
            elif "toolUse" in block:
                tu = block["toolUse"]
                content.append(ToolUseBlock(id=tu["toolUseId"], name=tu["name"], input=tu["input"]))

        usage = response.get("usage", {})
        return LLMResponse(
            stop_reason=response["stopReason"],
            content=content,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            model_id=self.model_id,
        )


# --- Offline / test implementation -----------------------------------------

Step = Callable[[list[dict[str, Any]]], LLMResponse]


class FakeLLMClient(LLMClient):
    """Deterministic offline stand-in. Driven by a list of `Step` callables,
    each producing the next turn's response given the conversation so far
    (so a step can, e.g., look at the most recent tool result and decide
    what to cite). Cycles are not expected; the agent loop ends once a step
    returns `stop_reason="end_turn"`.

    Turn index is derived from the conversation itself (the number of prior
    assistant turns in `messages`), not mutable instance state -- so one
    client instance is safe to reuse across multiple independent
    investigations (e.g. one FastAPI dependency override serving many
    requests), each starting its own `messages` list at turn 0.
    """

    def __init__(self, steps: list[Step], model_id: str = "fake-claude"):
        if not steps:
            raise ValueError("FakeLLMClient requires at least one step")
        self.steps = steps
        self.model_id = model_id

    def converse(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        turn_index = sum(1 for m in messages if m["role"] == "assistant")
        if turn_index >= len(self.steps):
            raise RuntimeError(
                f"FakeLLMClient exhausted its {len(self.steps)} scripted steps "
                "without the agent reaching end_turn -- the script is too short "
                "for this conversation, or the agent looped unexpectedly."
            )
        return self.steps[turn_index](messages)


def _last_tool_results(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract {tool_use_id: parsed_json_content} for the most recent user
    turn's tool results, keyed by id -- used by scripted steps to react to
    what a prior tool call actually returned."""
    if not messages or messages[-1]["role"] != "user":
        return {}
    results = {}
    for block in messages[-1]["content"]:
        if "toolResult" in block:
            tr = block["toolResult"]
            text = tr["content"][0]["text"]
            try:
                results[tr["toolUseId"]] = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                results[tr["toolUseId"]] = text
    return results


_TXN_ID_RE = re.compile(r'"transaction_id":\s*"([^"]+)"')


class AutoFakeLLMClient(LLMClient):
    """A `FakeLLMClient` that resolves which transaction it's investigating
    from the conversation itself (the transaction_id embedded in the first
    user turn's briefing JSON), rather than being constructed for one
    transaction ahead of time. This makes a single instance usable as a
    FastAPI dependency override or an evaluation-harness client serving
    many different transactions/requests without per-request setup.
    """

    def __init__(self, model_id: str = "fake-claude"):
        self.model_id = model_id
        self._scripts: dict[str, list[Step]] = {}

    def converse(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        first_text = messages[0]["content"][0]["text"]
        match = _TXN_ID_RE.search(first_text)
        if not match:
            raise ValueError("AutoFakeLLMClient could not find a transaction_id in the initial message")
        transaction_id = match.group(1)

        if transaction_id not in self._scripts:
            self._scripts[transaction_id] = default_investigation_script(transaction_id)

        turn_index = sum(1 for m in messages if m["role"] == "assistant")
        steps = self._scripts[transaction_id]
        if turn_index >= len(steps):
            raise RuntimeError(f"AutoFakeLLMClient exhausted its script for {transaction_id!r}")
        return steps[turn_index](messages)


def default_investigation_script(transaction_id: str) -> list[Step]:
    """A representative, deterministic script an investigation agent would
    plausibly follow: score -> gather cyber/graph evidence -> search policy
    -> narrate with citations -> (if high risk) propose a case. Used by
    tests and the offline evaluation harness so the whole tool-use loop in
    agent.py is exercised without a live Bedrock call.
    """
    from bankshield.investigation import data_access

    customer_id = data_access.get_transaction(transaction_id)["customer_id"]
    state: dict[str, Any] = {}

    def step_1_score(messages: list[dict[str, Any]]) -> LLMResponse:
        return LLMResponse(
            stop_reason="tool_use",
            content=[
                TextBlock(text="Let me start by pulling the model's risk score for this transaction."),
                ToolUseBlock(id="call_1", name="get_risk_score", input={"transaction_id": transaction_id}),
            ],
            input_tokens=800,
            output_tokens=60,
            model_id="fake-claude",
        )

    def step_2_gather(messages: list[dict[str, Any]]) -> LLMResponse:
        results = _last_tool_results(messages)
        risk = next(iter(results.values()), {})
        state["risk"] = risk
        return LLMResponse(
            stop_reason="tool_use",
            content=[
                TextBlock(text="Now let me check authentication history and network connections."),
                ToolUseBlock(id="call_2", name="get_auth_history", input={"customer_id": customer_id}),
                ToolUseBlock(id="call_3", name="get_graph_neighbors", input={"customer_id": customer_id}),
            ],
            input_tokens=1200,
            output_tokens=80,
            model_id="fake-claude",
        )

    def step_3_search_policy(messages: list[dict[str, Any]]) -> LLMResponse:
        risk_tier = state.get("risk", {}).get("risk_tier", "tier_3")
        query = (
            "account takeover failed logins new device pattern"
            if risk_tier in ("tier_1", "tier_2")
            else "ongoing transaction monitoring low risk"
        )
        return LLMResponse(
            stop_reason="tool_use",
            content=[
                ToolUseBlock(id="call_4", name="search_policy", input={"query": query}),
            ],
            input_tokens=1400,
            output_tokens=40,
            model_id="fake-claude",
        )

    def step_4_narrate(messages: list[dict[str, Any]]) -> LLMResponse:
        results = _last_tool_results(messages)
        search_result = next(iter(results.values()), {})
        citations = search_result.get("results", []) if isinstance(search_result, dict) else []
        risk = state.get("risk", {})
        risk_tier = risk.get("risk_tier", "tier_3")
        score = risk.get("risk_score", 0.0)
        cite = citations[0] if citations else {"doc_id": "POL-FRAUD-002", "section": "§3"}
        narrative = (
            f"Transaction {transaction_id} scored {score:.2f} ({risk_tier}) from the "
            f"transaction+cyber+graph model. This is model output, not a determination. "
            f"[{cite['doc_id']} {cite['section']}]"
        )
        if risk_tier == "tier_1":
            return LLMResponse(
                stop_reason="tool_use",
                content=[
                    TextBlock(text=narrative + " Recommending case creation for human review."),
                    ToolUseBlock(
                        id="call_5",
                        name="create_case",
                        input={
                            "transaction_id": transaction_id,
                            "disposition": "confirmed_fraud",
                            "summary": narrative,
                        },
                    ),
                ],
                input_tokens=1800,
                output_tokens=180,
                model_id="fake-claude",
            )
        return LLMResponse(
            stop_reason="end_turn",
            content=[TextBlock(text=narrative + " No consequential action recommended.")],
            input_tokens=1800,
            output_tokens=140,
            model_id="fake-claude",
        )

    def step_5_ack_pending(messages: list[dict[str, Any]]) -> LLMResponse:
        return LLMResponse(
            stop_reason="end_turn",
            content=[
                TextBlock(
                    text="Case proposal submitted and is pending human approval per POL-CASE-004 §3; "
                    "no case has been created yet."
                )
            ],
            input_tokens=2000,
            output_tokens=40,
            model_id="fake-claude",
        )

    return [step_1_score, step_2_gather, step_3_search_policy, step_4_narrate, step_5_ack_pending]
