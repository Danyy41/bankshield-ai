# BankShield AI — Phase 4 Architecture

Phase 4 converts the Phase 1–3 fraud/cyber/graph intelligence pipeline into
an AI-assisted investigation system: a service layer, a structured
investigation payload, a Bedrock/Claude agent grounded by RAG over
synthetic policy documents with a controlled tool set, and a human-approval
gate on consequential actions. This document explains the design decisions
behind that system and the tradeoffs deliberately deferred.

## Data flow

```
Phase 1-3 (unchanged)                Phase 4 (new, additive-only)
──────────────────────               ───────────────────────────────────────

transactions.csv  ┐
login_events.csv  ├─► data_access.py ─► InvestigationPayload (schemas.py)
graph_entities.csv┘         │                    │
xgboost_with_graph          │                    ▼
  _pipeline.joblib ─────────┘          agent.py: tool-use loop
                                                  │
data/policy_docs/*.md ─► policy_corpus.py         │  system prompt: ground every
                              │                    │  claim in a tool call; cite
                              ▼                    │  [DOC-ID §section]; create_case
                          rag.py (TF-IDF)  ◄────────┤  is consequential
                                                  │
                                    tools.py (6 tools, 1 gated) ◄──── llm_client.py
                                                  │                    (LLMClient:
                                    ┌─────────────┴──────────┐         Bedrock or
                                    │                         │         offline fake)
                          5 read-only tools         create_case (gated)
                          execute immediately              │
                                                             ▼
                                                   approvals.py: PendingApproval
                                                   (no case created yet)
                                                             │
                                          human decision (API: POST /approvals/{id}/decision)
                                                             │
                                                    ┌────────┴────────┐
                                               approved            rejected
                                                    │                  │
                                                    ▼                  ▼
                                          case_store.py: Case    (nothing created)

                                    InvestigationResult (payload + narrative +
                                    citations + tool_calls + pending_approvals +
                                    tokens/cost/latency)
                                                  │
                                                  ▼
                                     api/app.py (FastAPI) ─► analyst / evaluation harness
```

Every arrow into Phase 4 is read-only against Phase 1–3 outputs. Nothing in
`src/bankshield/investigation/` or `src/bankshield/api/` writes to
`data/raw`, `data/processed`, or `models/` — those directories, and every
Phase 1–3 module, are exactly as Phase 3 left them (verified by Phase 1–3's
own regression tests, which are unmodified and still pass).

## Key decisions and why

### 1. Service layer boundary: `data_access.py`, not scattered reads

Every place Phase 4 needs a transaction, a login event, a graph neighbor,
or the trained model goes through `investigation/data_access.py`. This is
the same principle Phase 1's README states for its own module boundaries
("`data`, `features`, `modeling`, `evaluation`... are intentionally where
later phases will plug in") — Phase 4 plugs in at exactly one new module,
not by reaching into Phase 1–3 internals from five different places.

`data_access.get_risk_score()` uses XGBoost's native
`booster.predict(dmatrix, pred_contribs=True)` for feature attribution —
an exact, per-transaction contribution to the model's margin, built into a
library the project already depends on. This was chosen over adding a
`shap` dependency: it's the same underlying math XGBoost's own contribution
API exposes, with one fewer dependency and no separate explainer object to
keep in sync with the trained model.

### 2. RAG: local TF-IDF, not an embeddings API

Two things need retrieval to be deterministic and network-free:

- **The test suite.** Phase 1–3's tests already run without network access;
  Phase 4's tests (and CI) shouldn't need one either.
- **The evaluation harness's citation-correctness metric**, which checks
  whether a cited passage was actually retrieved this run. That check only
  means something if retrieval is reproducible.

`investigation/rag.py` uses scikit-learn's `TfidfVectorizer` +
cosine similarity — a dependency the project already has for Phase 1's
preprocessing pipeline. `investigation/policy_corpus.py` chunks each policy
document by its own `## §N. Title` / `### §N.M Title` structure rather than
a fixed-size sliding window, because every synthetic document here is
written with one coherent idea per section — section boundaries are also
the right citation granularity.

**Deferred, deliberately:** a production deployment would likely swap this
for Bedrock Titan Embeddings + a vector store (OpenSearch, pgvector) for
semantic recall beyond keyword/n-gram overlap. The `PolicyRetriever`
interface (`search(query, top_k) -> [(chunk, score)]`) is small enough that
this is a backend swap, not a rewrite of anything that calls it.

### 3. `LLMClient`: Bedrock Converse shape, real + offline implementations

`investigation/llm_client.py` defines `LLMClient.converse(messages, system,
tools) -> LLMResponse` using Amazon Bedrock's Converse API message and tool
shapes directly, rather than inventing an internal format that would need
translating to and from Bedrock's on every call. `BedrockClaudeClient`
wraps `boto3`'s `bedrock-runtime` client; `FakeLLMClient` /
`AutoFakeLLMClient` are deterministic, script-driven implementations of the
exact same contract.

This sandbox has no AWS credentials, so `BedrockClaudeClient` is
implemented to the Converse API's documented shape but has not been
exercised against a live endpoint. Every test, the default evaluation run,
and `run_all.py`'s step 13 use the offline fake client instead — not as a
mock replacing real behavior, but as a second real implementation of the
same interface, which is what lets `agent.py`'s tool-use loop be verified
end-to-end without live infrastructure. `scripts/13_evaluate_investigations.py
--live` exercises the real backend when credentials are available.

### 4. Six tools, one gated

The tool surface (`investigation/tools.py`) mirrors what a human analyst is
told to gather in `POL-FRAUD-002 §2` (transaction facts, risk drivers, auth
history, network context) plus `search_policy` for citations. Five tools
are read-only and dispatch immediately. `create_case` is not: per
`POL-CASE-004 §3`, creating a case that names a customer is a
**consequential action**, and the policy is explicit that an AI assistant
"may propose... but the system must present that proposal to a human
reviewer and receive explicit, recorded approval before the action is
carried out."

That requirement is enforced structurally, not just requested in the
system prompt: `_tool_create_case()` never calls anything in
`case_store.py`. It calls `approvals.py`'s `ApprovalStore.request(...)`,
which returns a `PendingApproval` and registers an `on_approve` callback.
`case_store.create_from_approval()` is only ever invoked from inside
`ApprovalStore.decide(..., approved=True)` — a call that only happens from
a human-facing API endpoint
(`POST /approvals/{approval_id}/decision`). There is no function in this
codebase that goes from a tool call straight to a created `Case`; grep for
`create_from_approval` and its only caller is the approval decision path.
This is tested directly (`test_tools_and_approvals.py`,
`test_agent.py::test_run_investigation_never_creates_a_case_directly`,
and the full-workflow API test).

### 5. Citation grounding is enforced after generation, not trusted

The system prompt instructs the model to cite only retrieved passages, but
prompted behavior is not proof — models can still cite something they
remember rather than something they looked up. `agent.py`'s
`_extract_citations()` regex-matches every `[DOC-ID §section]`-shaped
mention in the narrative and keeps only the ones that **both** resolve to
a real chunk in the corpus **and** appear in the set of chunks actually
retrieved during that run (the payload-building search or an explicit
`search_policy` tool call). A citation to a real, existing policy section
the agent never retrieved is still dropped — citing something true from
memory is exactly the failure mode RAG grounding exists to prevent, not a
lesser offense than citing something false.

The evaluation harness's citation-correctness metric measures the same
thing at the aggregate level, and `test_agent.py`'s
`test_citations_are_filtered_to_only_actually_retrieved_chunks` exercises
it directly with a deliberately fabricated citation.

### 6. In-memory approval/case/investigation stores

`approvals.py`, `case_store.py`, and the API layer's investigation-results
cache are process-local, in-memory dataclasses. This is the same tradeoff a
demo/portfolio deployment makes elsewhere in this repo (e.g. the trained
models are files, not a model registry) — adequate for exercising the full
workflow and its tests, but **deliberately not** what a production
deployment would ship. A real deployment would back the approval queue and
case store with a durable, auditable store (e.g. DynamoDB, as suggested by
`POL-CASE-004 §4`'s approval-record requirements) so the queue survives a
restart and is queryable across instances.

## What Phase 5 tested

That red-teaming was out of scope for Phase 4 by design — see the
top-level task instructions — and is now done: Phase 5 (`src/bankshield/security/`,
`security/redteam_cases.yaml`, `scripts/run_redteam.py`) red-teamed exactly
the surfaces described above -- prompt injection via narrative text and API
fields, RAG/document poisoning, attempts to make `create_case` skip the
approval gate, and citation-grounding bypass attempts -- and fixed one real
gap it found (the citation check verified corpus existence but not
retrieval-this-run). See
[`reports/phase5_redteam_report.md`](phase5_redteam_report.md) and the
README's Phase 5 section for the full results and defenses.
