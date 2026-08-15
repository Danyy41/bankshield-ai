# BankShield AI

BankShield AI is a portfolio project demonstrating production-style AI engineering
across financial crime detection, cybersecurity, and applied ML. It is being built
in phases, each one a complete, working slice rather than a stub.

**This repository currently implements Phase 1, Phase 2, Phase 3, and Phase 4.**

## Phase 1: fraud-detection ML baseline

A clean, honest baseline: synthetic banking transaction data, careful feature
engineering, a leakage-safe train/test split, two classifiers (Logistic
Regression and XGBoost), and evaluation metrics chosen for a rare-event
classification problem rather than misleading accuracy.

No AWS, no LLMs, no RAG, no agents, no frontend yet — those are later
phases, and this project is deliberately structured so they can be added
without reworking Phase 1 (see [Roadmap](#roadmap)). Phase 1's data,
models, and metrics are untouched by Phases 2 and 3 below — this is
verified by regression tests and by re-running the full pipeline and
diffing the outputs (model file hashes and metrics JSON come out
byte-identical) after every later-phase change.

### What's in the data

`scripts/01_generate_data.py` generates ~50,000 synthetic transactions across
8,000 customers over a 120-day window (`src/bankshield/data_generation.py`).
Fraud is rare (~1.5%) and generated as a **two-stage process**:

1. Behavioural features (device used, transaction country, hour, amount,
   category, ...) are drawn from realistic population-level base rates that
   do not depend on the eventual label.
2. A fraud probability is then computed from those realized features via a
   logistic combination of risk signals — including a few genuine
   **interaction effects** (e.g. a new device used at night, or a
   cross-border transfer into a high-risk category, are riskier together
   than the sum of their individual effects) — plus Gaussian noise, and the
   label is sampled from that probability.

This makes fraud **statistically distinguishable but not trivially
separable**: every risk factor raises the fraud rate, but none of them
comes close to perfectly predicting it (verified in `tests/`).

| Field | Description |
|---|---|
| `transaction_id`, `customer_id` | identifiers |
| `timestamp` | when the transaction occurred |
| `amount` | transaction amount, category- and customer-dependent |
| `merchant_category` | e.g. grocery, travel, wire_transfer, crypto_exchange |
| `home_country`, `country`, `country_mismatch` | customer's home country vs. where the transaction happened |
| `device_id`, `new_device` | device used, and whether it's new for this customer |
| `ip_address` | transaction IP |
| `account_age_days` | age of the account at transaction time |
| `transaction_velocity_24h` | this customer's transaction count in the trailing 24h |
| `new_beneficiary` | whether a transfer/withdrawal target is new for this customer |
| `hour_of_day`, `day_of_week`, `is_night` | timing features |
| `amount_to_avg_ratio` | this transaction's amount vs. the customer's own prior average |
| `is_fraud` | target label |

Every one of these engineered features (`new_device`, `transaction_velocity_24h`,
`amount_to_avg_ratio`, ...) is computed **causally**: by walking each
customer's transactions in timestamp order and only looking at that
customer's own past. No feature ever depends on information that wouldn't
be available at the moment the transaction occurred — the first place
leakage could otherwise be introduced.

### Pipeline

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # includes fastapi/uvicorn/pydantic/boto3 for Phase 4

python scripts/run_all.py          # runs the full Phase 1 + 2 + 3 + 4 pipeline (steps 1-13)
uvicorn bankshield.api.app:app --reload --app-dir src   # serve the Phase 4 API locally
```

Or step by step (Phase 1; see [Phase 2](#phase-2-cyber-authentication-telemetry) for steps 7-9):

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_generate_data.py` | Generate synthetic transactions → `data/raw/transactions.csv` |
| 2 | `scripts/02_run_eda.py` | Exploratory analysis → `reports/figures/`, `reports/eda_summary.md` |
| 3 | `scripts/03_split_data.py` | Chronological train/test split → `data/processed/` |
| 4 | `scripts/04_train_baseline.py` | Train + evaluate Logistic Regression → `models/baseline_logreg_pipeline.joblib` |
| 5 | `scripts/05_train_xgboost.py` | Train + evaluate XGBoost → `models/xgboost_pipeline.joblib` |
| 6 | `scripts/06_compare_models.py` | Compare both models → `reports/model_comparison.md` |

Run the test suite (data-generation and split sanity checks) with:

```
pip install -r requirements-dev.txt
pytest tests/
```

### Splitting without leakage

The train/test split is **chronological**, not a random shuffle: the
earliest 80% of transactions (by timestamp) train the model, the most
recent 20% test it. A random shuffle would let a customer's March
transaction sit in the training set while their February transaction sits
in the test set — implicitly letting the model see "the future" relative to
what it's being tested on. In production a fraud model only ever scores
transactions that happened after everything it was trained on, so the
evaluation should reflect that. It also means the same customer can
legitimately appear in both splits (their early transactions train, their
later ones test) — what never happens is the same *transaction* appearing
in both.

### Results

Test set: 10,000 transactions, 1.38% fraud prevalence.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.828 | 0.055 | 0.710 | 0.102 | 0.821 | 0.136 |
| XGBoost | 0.979 | 0.229 | 0.217 | 0.223 | **0.826** | **0.155** |

(Exact numbers regenerate identically on re-run — the whole pipeline is
seeded. Current values are also saved in `reports/metrics/*.json`.)

XGBoost modestly outperforms the linear baseline on both threshold-independent
metrics (ROC-AUC, PR-AUC), consistent with the data containing real
non-additive interaction effects that a linear model can only approximate.
The two models sit at very different points on the precision/recall
trade-off at their default decision threshold: `class_weight="balanced"`
pushes the logistic regression to catch 71% of fraud at the cost of very
low precision (5.5% — 1,678 false alarms to catch 98 fraud cases), while
XGBoost's more moderate `scale_pos_weight` trades some recall for far fewer
false positives. Neither is "more correct" in the abstract — the right
operating point is a business decision (cost of a missed fraud vs. cost of
a false alarm), which is why this project reports the full metric suite
and the PR/ROC curves (`reports/figures/roc_pr_comparison.png`, generated
by `scripts/06_compare_models.py`) instead of picking one number and one
threshold.

### Why accuracy alone is misleading

Fraud is rare: only 1.38% of test transactions are fraudulent. A model that
predicts "legitimate" for every single transaction — catching zero fraud —
would score **98.62% accuracy**, beating both real models trained here on
that metric alone. Accuracy weighs every prediction equally, so under this
much class imbalance it's almost entirely determined by performance on the
uninteresting 98.6% majority class. That's why this project always reports:

- **Precision** — of the transactions flagged, how many were actually fraud
  (controls false-alarm cost: blocked legitimate customers, review workload).
- **Recall** — of the actual fraud, how much was caught (the number a bank
  cares about most when limiting losses).
- **F1** — the harmonic mean of the two, useful when there's no single
  dominant business cost.
- **ROC-AUC** — ranking quality across all thresholds, but can look
  artificially strong under heavy imbalance since the false positive *rate*
  stays low even when false positives significantly outnumber true
  positives in absolute count.
- **PR-AUC** — the more honest summary metric here: a random classifier's
  PR-AUC baseline equals the fraud prevalence itself (~1.4%, not 0.5), so it
  directly reflects how hard the imbalance makes precision to achieve.
- **Confusion matrix** — the actual counts behind all of the above.

Full write-up with worked numbers: [`reports/model_comparison.md`](reports/model_comparison.md).

### Exploratory analysis

See [`reports/eda_summary.md`](reports/eda_summary.md) for class balance,
fraud rate by hour/category, and risk-factor lift numbers (e.g.
transactions with a new beneficiary are roughly 7-11x more likely to be
fraud than the baseline rate). Running `scripts/02_run_eda.py` additionally
saves plots (class balance, amount distributions, fraud rate by hour and
category, risk-factor lift, correlation matrix) to `reports/figures/`.

### Project structure

```
bankshield-ai/
├── src/bankshield/          # importable package — the actual logic
│   ├── config.py            # paths, column lists, run parameters (Phase 1 + 2 + 3 + 4)
│   ├── data_generation.py   # synthetic transaction generator (Phase 1)
│   ├── features.py          # train/test split + preprocessing pipeline
│   ├── modeling.py          # model definitions (LogReg, XGBoost)
│   ├── evaluation.py        # metrics, confusion matrix, PR/ROC plots
│   ├── eda.py                # exploratory analysis plots
│   ├── auth_generation.py   # synthetic login/session event generator (Phase 2)
│   ├── cyber_features.py    # causal login-history -> transaction feature merge (Phase 2)
│   ├── graph_generation.py  # beneficiary reconstruction + mule-ring injection (Phase 3)
│   ├── graph_features.py    # causal graph-relationship -> transaction feature merge (Phase 3)
│   ├── investigation/       # Phase 4: AI-assisted investigation
│   │   ├── schemas.py        # structured investigation payload (Pydantic)
│   │   ├── data_access.py    # read-only access to Phase 1-3 models/data
│   │   ├── policy_corpus.py  # parses data/policy_docs/ into citable chunks
│   │   ├── rag.py            # TF-IDF retrieval over the policy corpus
│   │   ├── llm_client.py     # LLMClient interface: Bedrock + offline fake
│   │   ├── tools.py          # controlled tool registry + dispatch
│   │   ├── approvals.py      # human-approval gate for consequential actions
│   │   ├── case_store.py     # cases (created only after approval)
│   │   ├── agent.py          # tool-use loop -> cited analyst explanation
│   │   └── evaluation.py     # citation/faithfulness/cost/latency eval harness
│   └── api/
│       └── app.py            # FastAPI service layer (Phase 1-4 endpoints)
├── scripts/                 # thin, numbered, run-in-order entry points (01-06 Phase 1, 07-09 Phase 2, 10-12 Phase 3, 13 Phase 4)
├── tests/                   # data-generation, split, cyber-, graph-, and investigation-layer sanity checks
├── data/
│   ├── {raw,processed}/     # generated CSVs (gitignored, regenerate via scripts)
│   └── policy_docs/         # synthetic AML/fraud/ATO/case/KYC policy docs (versioned, Phase 4)
├── models/                  # saved joblib pipelines (gitignored, regenerate via scripts)
└── reports/
    ├── metrics/              # evaluation JSON (versioned)
    ├── figures/              # EDA + evaluation plots (gitignored, regenerate via scripts)
    └── *.md                  # EDA, model-comparison, and Phase 4 eval/architecture write-ups (versioned)
```

Preprocessing (`StandardScaler` + `OneHotEncoder`) lives inside the same
`sklearn.Pipeline` as each classifier, so `models/*.joblib` is a single
self-contained artifact — load it and call `.predict()` on a raw feature
DataFrame, no separate preprocessing step to keep in sync.

`src/bankshield`'s module boundaries (`data`, `features`, `modeling`,
`evaluation`) are intentionally where later phases will plug in — new data
sources feed `data`, new engineered signals feed `features`, new model
types feed `modeling` — without needing to restructure what's already here.

## Phase 2: cyber-authentication telemetry

Adds synthetic login/session events for the same customers and asks a
concrete question: **does knowing about login behaviour, in addition to
the transaction itself, improve fraud detection?**

### Design: correlated with fraud, without touching Phase 1

Phase 1's `is_fraud` labels are fixed and Phase 2 never modifies them —
login events are generated by *reading* the existing `transactions.csv`,
not the other way around. Every transaction gets a "pre-auth" login
session shortly before it:

- **Fraudulent transactions** are preceded by an account-takeover (ATO)
  login pattern most of the time (75%): a burst of 2-6 failed attempts
  from unfamiliar devices/countries, ending in one success from a new
  device, usually from an unusual country.
- **Legitimate transactions** almost always get a single ordinary login
  from a familiar device/country, but a small fraction (3%) still show a
  benign ATO-like burst — a mistyped password or a genuinely new phone,
  with nothing fraudulent behind it. This keeps the signal statistical,
  not a deterministic tell.
- A handful of standalone "routine" logins per customer (e.g. balance
  checks) round out the picture.

Because compromise-then-transact is the actual real-world causal order,
this produces genuine, temporally-prior signal for a model to find — not
fabricated correlation. `src/bankshield/auth_generation.py` generates the
events; `src/bankshield/cyber_features.py` aggregates each customer's
login history onto their transactions, using **only logins strictly
before the transaction's own timestamp** (verified by an exact-value
unit test in `tests/test_auth_generation.py`).

| Field | Description |
|---|---|
| `customer_id`, `timestamp` | who, when |
| `device_id`, `ip_address`, `country` | where the login came from |
| `login_success` | whether the attempt succeeded |
| `new_device`, `new_location` | unfamiliar device / country for this customer |
| `session_duration_seconds` | length of the authenticated session (0 for failed attempts) |
| `session_id` | groups a burst of attempts + outcome into one session |
| `is_ato_pattern` | generation-time diagnostic marker (never used as a model feature) |

Engineered per-transaction cyber-risk features (all causal, all prefixed
`cyber_`): `failed_logins_1h`, `login_count_24h`,
`minutes_since_last_login`, `new_device_recent`, `unusual_country_recent`,
and a composite `recent_suspicious_auth` (several recent failures *and* a
suspicious final success — the actual ATO shape, not either signal alone).

### Pipeline (continues from Phase 1)

| Step | Script | What it does |
|---|---|---|
| 7 | `scripts/07_generate_login_events.py` | Generate login events from `transactions.csv` → `data/raw/login_events.csv` |
| 8 | `scripts/08_add_cyber_features.py` | Merge cyber features onto transactions, re-split → `data/processed/train_with_cyber.csv` / `test_with_cyber.csv` |
| 9 | `scripts/09_train_compare_cyber.py` | Train transaction-only vs. transaction+cyber XGBoost, compare → `reports/phase2_comparison.md` |

Step 8 re-splits the enriched dataset with the exact same
`time_based_split` used in Phase 1, so `train_with_cyber.csv` /
`test_with_cyber.csv` contain identical rows (by `transaction_id`) to
Phase 1's `train.csv` / `test.csv` — the only difference is the appended
`cyber_*` columns. Step 9 verifies this and re-trains the transaction-only
model on those rows as a control; its metrics come out identical to
Phase 1's original numbers, confirming the comparison is apples-to-apples
and Phase 1 is unaffected.

### Results: does cyber telemetry help?

Test set: 10,000 transactions, 1.38% fraud prevalence. Both models use
identical rows, hyperparameters, and `scale_pos_weight` — the only
difference is whether `cyber_*` features are available.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Transaction-only | 0.229 | 0.217 | 0.223 | 0.826 | 0.155 |
| Transaction + cyber | 0.432 | 0.645 | 0.517 | 0.925 | 0.551 |

**Yes — substantially.** PR-AUC more than triples (0.155 → 0.551), ROC-AUC
rises from 0.826 to 0.925, F1 more than doubles, and recall jumps from
22% to 64% at a comparable false-positive rate. In the trained model's
feature importances, `cyber_recent_suspicious_auth` and
`cyber_failed_logins_1h` are the top two features by a wide margin —
exactly the ATO shape the generator was designed around. Full write-up:
[`reports/phase2_comparison.md`](reports/phase2_comparison.md).

This result should be read for what it is: strong evidence that *if*
login telemetry carries an ATO signal like this, it's highly valuable —
not a claim about the exact multiplier a real deployment would see, since
the synthetic correlation strength (75% injection rate, 3% false-alarm
rate) was a deliberate design choice, not fit to real data.

### Tests

`tests/test_auth_generation.py` adds: login event schema/uniqueness,
ATO-pattern rate correlating with fraud (but staying strictly between 0
and 1 — not a trivial tell), an exact-value causality check that cyber
features never see a login that happens after the transaction, cyber
feature correlation with fraud on generated data, and a regression guard
that Phase 1's `get_X_y`/`build_preprocessor` still behave identically
when called with no arguments (both gained optional parameters for
Phase 2 but default to the original Phase 1 behavior).

## Phase 3: graph-based financial crime intelligence

Adds relationships *between* customers — shared devices, IPs, and money-
transfer beneficiaries — and asks a further question: **once a model
already has transaction and cyber-telemetry features, does knowing who's
connected to whom add anything more?**

### Design: simulating mule rings, without touching Phase 1/2

Phase 1 never persisted an actual beneficiary ID (only a `new_beneficiary`
boolean), and independently-randomized device IDs/IPs essentially never
collide by chance — so there's no organic account-to-account structure to
mine. Phase 3 builds it, in a new module (`graph_generation.py`) that only
ever reads Phase 1/2's existing, fixed data:

1. **Reconstructs beneficiary IDs** consistent with the existing
   `new_beneficiary` flag — a customer's first "new beneficiary"
   transaction gets a fresh ID, later non-new ones reuse one of their own
   prior IDs. These are new synthetic labels generated here, not values
   recovered from Phase 1's internal state (Phase 1 never stored them).
2. **Injects mule rings**: ~25 rings of 3-7 customers, membership skewed
   toward (never limited to) customers who already have a fraudulent
   transaction, each ring sharing a small pool of devices/IPs/beneficiary
   IDs across a fraction of members' transactions (biased toward their
   fraudulent ones, never all of them). Ring members end up with roughly
   an 11% fraud rate versus ~1.3% for everyone else — real, useful
   structure, not a deterministic tell.

Everything lands in new columns (`graph_device_id`, `graph_ip_address`,
`beneficiary_id`) that default to a transaction's own real device/IP — the
`device_id`/`ip_address`/`new_device` columns Phase 1/2 rely on are
untouched.

### Causal graph construction — the leakage-critical part

Phase 1/2's causal features are computed per customer in isolation
(walking one customer's own history forward in time). A graph is
different: it connects customers *to each other*, so `graph_features.py`
instead makes a **single chronological pass over all transactions**,
maintaining incremental state (who has used which device/IP/beneficiary
so far, and which customers are known to have committed fraud so far).
For every transaction, in timestamp order: compute its features from that
state, *then* fold the transaction's own device/IP/beneficiary usage and
fraud outcome into the state for the future. A transaction never sees its
own contribution or anything that happens later.

This surfaced a real bug during development: the edge-forming step
initially didn't exclude a customer's own prior usage from their "shared
with" set, so a customer reusing their own device created a false
self-loop that inflated their own neighbor count. An exact-value unit
test (`test_graph_features_no_future_leakage`) caught it before it shipped
— now checked into the test suite as a permanent regression guard.
A second, unrelated determinism bug was also caught this way: ring
membership was iterated from a Python `set`, whose order is randomized
per-process by design (`PYTHONHASHSEED`), which silently reshuffled which
random draws went to which ring member on every run. Fixed by iterating a
sorted, order-stable sequence instead — verified by regenerating twice and
diffing the output.

Features (all causal, all prefixed `graph_`): `shared_device_count`,
`shared_ip_count`, `beneficiary_connectivity` (other customers who share
this device/IP/beneficiary, before now), `suspicious_neighbor_count`
(known neighbors with prior fraud), and `account_network_risk` (fraud
rate among those neighbors).

### Pipeline (continues from Phase 1/2)

| Step | Script | What it does |
|---|---|---|
| 10 | `scripts/10_generate_graph_relationships.py` | Reconstruct beneficiary IDs, inject mule rings → `data/raw/graph_entities.csv` |
| 11 | `scripts/11_add_graph_features.py` | Merge graph features onto the cyber-enriched transactions, re-split → `train_with_graph.csv` / `test_with_graph.csv` |
| 12 | `scripts/12_train_compare_graph.py` | Train transaction+cyber vs. transaction+cyber+graph XGBoost, compare → `reports/phase3_comparison.md` |

Step 11 re-splits with the same `time_based_split` used throughout, so row
membership matches Phase 1/2's splits exactly (verified in-script, same as
Phase 2).

### Results: does graph intelligence help further?

Test set: 10,000 transactions, 1.38% fraud prevalence. Both models use
identical rows, hyperparameters, and `scale_pos_weight`.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Transaction + cyber | 0.432 | 0.645 | 0.517 | 0.925 | 0.551 |
| Transaction + cyber + graph | 0.428 | 0.623 | 0.507 | **0.937** | **0.572** |

**Yes, further — modestly, and with an honest caveat.** Both
threshold-independent metrics improve (ROC-AUC +0.013, PR-AUC +0.021),
meaning the model got better at ranking fraud above legitimate
transactions. But at the fixed default 0.5 threshold, precision/recall/F1
actually dip slightly — adding features shifts the model's predicted-
probability distribution, so the cutoff tuned for transaction+cyber isn't
automatically optimal for transaction+cyber+graph. That's a threshold-
calibration artifact, not evidence against the features: a deployment
would re-tune the decision threshold before shipping either model, not
compare them at a cutoff neither was chosen for. In the trained model's
feature importances, `graph_suspicious_neighbor_count` and
`graph_shared_device_count` both place in the top 12 — real, if smaller
than cyber telemetry's, since most of the detection gain over
transaction-only was already captured in Phase 2. Full write-up:
[`reports/phase3_comparison.md`](reports/phase3_comparison.md).

### Tests

`tests/test_graph_generation.py` adds: graph-entity schema/uniqueness,
ring membership correlating with fraud without being deterministic,
exact-value beneficiary-ID reconstruction consistency, the exact-value
causal leakage test described above (including the specific case that
caught the self-loop bug — a customer reusing their own device before
anyone else touches it), graph feature correlation with fraud on
generated data, and row/column preservation checks.

## Phase 4: AI-assisted financial crime investigation

Phases 1–3 produce risk scores and features. Phase 4 turns that pipeline
into something an analyst can actually work with: a service layer that
exposes the model's outputs, an LLM-powered agent (Amazon Bedrock + Claude)
that gathers evidence through controlled tools and explains *why* an alert
fired with citations to policy, and a human-approval gate so the agent can
never take a consequential action on its own. Phase 1–3 code and data are
untouched — everything below only *reads* what those phases already
produce.

### Why this design

**A clean service layer, not a rewrite.** `src/bankshield/investigation/`
is a new package that reads Phase 1–3's trained model and CSVs through one
module (`data_access.py`) and never modifies them. Explanations for a
transaction's risk score use XGBoost's own `pred_contribs` output — exact,
per-transaction feature attribution built into the library already used —
instead of adding a `shap` dependency.

**RAG over a local TF-IDF index, not an embeddings API.** `data/policy_docs/`
holds six synthetic AML/fraud/ATO/case-management/KYC/mule-network policy
documents (`POL-XXX-NNN`, section-numbered for citation, e.g.
`POL-ATO-003 §2.1`). Retrieval (`investigation/rag.py`) uses scikit-learn's
`TfidfVectorizer` — a dependency the project already has — chunked by the
documents' own section structure rather than a fixed-size window, so a
citation reads naturally and retrieves cleanly. This keeps retrieval
deterministic and network-free, which matters twice over: the test suite
and the evaluation harness both need to run the same way every time,
without AWS credentials.

**An `LLMClient` interface, with Bedrock and a scripted offline
implementation.** `investigation/llm_client.py` defines the contract as
Amazon Bedrock's Converse API shape directly (no extra translation layer),
with two implementations: `BedrockClaudeClient` (real, via `boto3`) and
`FakeLLMClient` / `AutoFakeLLMClient` (deterministic, offline). This sandbox
has no AWS credentials, so every test and the default evaluation run
against the fake client — it is a legitimate implementation of the same
tool-use loop, not a mock, which is what lets `agent.py`'s orchestration be
exercised end-to-end in CI. Point the API at `BedrockClaudeClient()` (the
default `get_llm_client` dependency) to use the real model; see
[Running against real Bedrock](#running-against-real-bedrock).

**Six controlled tools, one of them gated.** The agent can call
`get_transaction`, `get_risk_score`, `get_auth_history`,
`get_graph_neighbors`, and `search_policy` freely — all read-only. The
sixth, `create_case`, is different: per `POL-CASE-004 §3`, naming a
customer in a case is a *consequential action*. Calling the tool never
creates anything — it files a `PendingApproval`
(`investigation/approvals.py`) and returns that pending state to the agent,
which must report it as pending, not as done. Only a separate, explicit
`POST /approvals/{id}/decision` call — a human reviewer's decision — can
turn a proposal into a `Case`. This is enforced in code (there is no
function that goes straight from a tool call to a created case), not just
requested in the system prompt.

### Structured investigation payload

`investigation/schemas.py` defines the grounding context every
investigation is built from: `TransactionRisk` (score, tier, top
contributing features with signed contributions), `CyberSignals`,
`GraphSignals`, raw `TransactionEvidence`, recent `AuthEvent`s,
`GraphNeighbor`s, and `PolicyCitation`s retrieved for the alert —
assembled into one `InvestigationPayload` before the agent's tool-use loop
even starts. The full run (payload + narrative + citations + every tool
call + any pending approval + token/cost/latency accounting) is an
`InvestigationResult`.

### API surface

`src/bankshield/api/app.py` (FastAPI):

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check — `{"status": "ok"}` |
| `GET /demo/sample-transactions` | 5 (or `?n=`) valid transaction_ids with risk score/tier — see [below](#trying-the-demo) |
| `GET /transactions/{id}`, `/transactions/{id}/risk` | Phase 1–3 passthrough |
| `GET /customers/{id}/auth-history`, `/graph-neighbors` | Phase 2/3 passthrough |
| `POST /policy/search` | RAG search over the policy corpus |
| `POST /investigations` | Run the agent on a transaction_id |
| `GET /investigations/{id}` | Retrieve a prior run |
| `GET /approvals`, `POST /approvals/{id}/decision` | The human-approval queue |
| `GET /cases`, `GET /cases/{id}` | Cases created only after approval |

### Trying the demo

The synthetic dataset's transaction_ids aren't documented anywhere a caller
would otherwise see them, so `GET /demo/sample-transactions` hands out a
few valid, real ones to get started against a freshly deployed instance —
by default a demo-friendly mix spread across risk tiers (up to 2 tier_1, 2
tier_2, 1 tier_3), not just whatever sorts first:

```
curl http://localhost:8000/demo/sample-transactions
# {"transactions": [
#   {"transaction_id": "...", "risk_score": 0.986, "risk_tier": "tier_1"},
#   {"transaction_id": "...", "risk_score": 0.005, "risk_tier": "tier_3"},
#   ...
# ]}

curl http://localhost:8000/investigations \
  -X POST -H 'content-type: application/json' \
  -d '{"transaction_id": "<one of the ids above>"}'
```

The underlying scan (`investigation.data_access.sample_transactions`) is
cached after its first call — tier_1 alerts are rare in this dataset (see
POL-AML-001 §2.1), so finding a couple can mean scoring a few hundred
candidates; the endpoint pays that cost once per process, not per request.

### Analyst-facing explanations, grounded and cited

The agent's system prompt requires every factual claim to come from a tool
call (risk score before describing risk, auth history before describing
login behavior, policy search before citing policy) and every policy
citation to reference a passage it actually retrieved, formatted
`[DOC-ID §section]`. `agent.py` then filters the narrative's citations down
to ones that both exist in the corpus and were actually retrieved during
that run — a citation to a real section the agent never looked up is
dropped, not trusted.

### Deployment mode: `BANKSHIELD_LLM_MODE`

`api/app.py`'s `get_llm_client` dependency — which `LLMClient` backs
`POST /investigations` — is selected by the `BANKSHIELD_LLM_MODE`
environment variable, not a code change:

| `BANKSHIELD_LLM_MODE` | Backend | Requires |
|---|---|---|
| `offline` (default) | `AutoFakeLLMClient` — deterministic, scripted | Nothing. Safe default for a public demo deployment. |
| `bedrock` | `BedrockClaudeClient` — real Amazon Bedrock Converse API for `anthropic.claude-opus-5` (`config.BEDROCK_MODEL_ID_DEFAULT`, `config.BEDROCK_REGION_DEFAULT` — both overridable in `config.py`) | AWS credentials with `bedrock:InvokeModel` for that model in your account/region |

Leaving `BANKSHIELD_LLM_MODE` unset means the deployed API serves real
requests end-to-end (transactions, risk scores, investigations, the
approval workflow) with **zero configuration and no AWS dependency** —
this is the intended default for a publicly reachable demo. Nothing in
this repo has been run against a live Bedrock endpoint (no AWS credentials
in this development environment), so treat `bedrock` mode's request/response
handling as implemented-to-spec rather than field-tested — verify it in
your own AWS account before relying on it.

**Deployment commands:**

```
# Public demo (default) — offline, no AWS credentials needed
uvicorn bankshield.api.app:app --app-dir src --host 0.0.0.0 --port 8000

# Explicit offline mode (equivalent to the default)
BANKSHIELD_LLM_MODE=offline uvicorn bankshield.api.app:app --app-dir src --host 0.0.0.0 --port 8000

# Real Bedrock/Claude backend — requires AWS credentials in the environment
BANKSHIELD_LLM_MODE=bedrock uvicorn bankshield.api.app:app --app-dir src --host 0.0.0.0 --port 8000

# Local development, auto-reload
uvicorn bankshield.api.app:app --reload --app-dir src
```

An unrecognized `BANKSHIELD_LLM_MODE` value raises a clear `RuntimeError`
(`Invalid BANKSHIELD_LLM_MODE=... ; expected 'offline' or 'bedrock'.`) the
first time `get_llm_client` is resolved, rather than silently falling back
to a default. The evaluation harness has its own, independent `--live` flag
for the same offline/Bedrock choice:

```
python scripts/13_evaluate_investigations.py           # offline (default)
python scripts/13_evaluate_investigations.py --live    # real Bedrock
```

### Evaluation

`scripts/13_evaluate_investigations.py` runs the agent over a golden set of
transactions spread across all three risk tiers (`POL-AML-001 §2.1`) and
reports:

- **Citation correctness** — of every `[DOC-ID §section]` citation, the
  fraction that both exist in the corpus and were actually retrieved this
  run (not just present somewhere in the corpus — an ungrounded-but-real
  citation still counts as wrong).
- **Evidence faithfulness** — whether numeric/boolean claims in the
  narrative (the stated risk score, ATO-pattern claims) match the actual
  payload values.
- **Tool-call success rate** — fraction of tool calls that didn't error.
- **Latency** and **estimated cost** — wall-clock time and a list-price
  token-cost estimate per investigation.

Offline mode (default, `AutoFakeLLMClient`, no AWS credentials) is what
runs in CI and via `run_all.py`; pass `--live` to evaluate the real Bedrock
backend instead. Current offline numbers (13 investigations, 4–5 per tier):

| Metric | Value |
|---|---|
| Citation correctness | 1.000 |
| Evidence faithfulness | 1.000 |
| Tool-call success rate | 1.000 |
| Mean latency | ~80 ms |

These numbers describe the scripted offline agent's self-consistency (it
always cites what it just retrieved and states the score it just looked
up) — they are a regression guard on the harness and the grounding/citation
mechanism, not a claim about a live LLM's behavior. A `--live` run against
real Bedrock is expected to score lower on citation correctness and
evidence faithfulness, since a real model can still misquote or omit a
citation despite the grounding; that gap is exactly what this harness
exists to measure. Full write-up: [`reports/phase4_eval.md`](reports/phase4_eval.md).

### Tests

`tests/test_policy_corpus_and_rag.py`, `test_data_access.py`,
`test_llm_client.py`, `test_tools_and_approvals.py`, `test_agent.py`,
`test_api.py`, and `test_evaluation.py` add: policy-document chunking and
citation-label correctness, RAG retrieval relevance, data-access causality
(auth history drill-down never surfaces a login at or after the queried
transaction's own timestamp) and neighbor-graph symmetry, the risk-score
tool's feature attribution, full tool dispatch, the human-approval gate
(a `create_case` call never creates a case directly; only an explicit
approval decision does — tested at both the tool layer and the FastAPI
layer), end-to-end agent runs on real Tier 1/2/3 transactions from the
generated dataset, citation filtering against a deliberately fabricated
citation, and the evaluation harness's determinism and aggregation. None of
this touches or reruns Phase 1–3's own tests, which continue to pass
unmodified.

### Architecture write-up

See [`reports/phase4_architecture.md`](reports/phase4_architecture.md) for
a fuller discussion of the design decisions above, including the tradeoffs
deliberately deferred (a production vector DB and embeddings backend for
RAG, a durable approval/case store instead of in-memory, and Phase 5's
planned adversarial testing of the agent itself).

## Roadmap

Phase 1, 2, 3, and 4 (this repo) are the classical ML + cyber-telemetry +
graph-intelligence foundation, plus the AI-assisted investigation layer on
top of it. Planned next:

- **Phase 5** — AI-security testing of the Phase 4 system (prompt
  injection against the agent and its tools, attempts to bypass the
  human-approval gate, model evasion, adversarial robustness).

## License

Portfolio project — no license specified.
