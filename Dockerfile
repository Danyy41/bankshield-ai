# syntax=docker/dockerfile:1.7
#
# BankShield AI -- Phase 4 investigation API (bankshield.api.app:app).
#
# Two stages:
#   builder  installs the pinned stack and runs the seeded Phase 1-3
#            pipeline steps that produce the artifacts the API reads (the
#            graph-enriched transactions, login events, graph entities, and
#            the XGBoost+graph model). The pipeline is deterministic on the
#            pinned stack, so every build bakes in the same data and model.
#   runtime  slim image with only the runtime dependencies, the source, the
#            policy corpus, and those artifacts. Runs as a non-root user.
#
# Build:   docker build -t bankshield-ai .
# Run:     docker run --rm -p 8000:8000 bankshield-ai
#
# Behind a TLS-intercepting proxy, pass its CA bundle to pip as a build
# secret (it is never written into an image layer):
#   docker build --secret id=pip_cert,src=/path/to/ca-bundle.crt -t bankshield-ai .

# python:3.11-slim, pinned by digest -- Python 3.11 is the reference
# environment requirements-lock.txt was resolved for.
ARG PYTHON_IMAGE=python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# No apt packages needed: the xgboost and scikit-learn wheels bundle their
# own OpenMP runtime (xgboost.libs/, scikit_learn.libs/).
WORKDIR /build

COPY requirements.txt requirements-lock.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=secret,id=pip_cert,required=false \
    if [ -s /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi; \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt -c requirements-lock.txt \
    # NCCL is only used for multi-GPU training; xgboost loads it lazily, and
    # this image is CPU-only. Dropping it saves a few hundred MB.
    && /opt/venv/bin/pip uninstall -y nvidia-nccl-cu12

ENV PATH=/opt/venv/bin:$PATH

COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
RUN mkdir -p models reports/metrics reports/figures \
    && for step in 01_generate_data 03_split_data 07_generate_login_events \
                   08_add_cyber_features 10_generate_graph_relationships \
                   11_add_graph_features 12_train_compare_graph; do \
         python "scripts/${step}.py" || exit 1; \
       done

# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="BankShield AI" \
      org.opencontainers.image.description="AI-assisted financial crime investigation API (offline mode by default)" \
      org.opencontainers.image.source="https://github.com/Danyy41/bankshield-ai"

RUN groupadd --system --gid 10001 bankshield \
    && useradd --system --uid 10001 --gid bankshield --no-create-home --home-dir /app bankshield

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    # Offline (AutoFakeLLMClient) unless explicitly overridden at `docker run`:
    # "bedrock" also needs AWS credentials with bedrock:InvokeModel.
    BANKSHIELD_LLM_MODE=offline \
    # Durable state (users, approvals, cases, audit log) on the /app/var
    # volume. Point this at PostgreSQL (e.g. RDS) for a real deployment.
    # BANKSHIELD_JWT_SECRET is deliberately not set here: supply it at run
    # time (e.g. from AWS Secrets Manager), never in the image.
    BANKSHIELD_DATABASE_URL=sqlite:////app/var/bankshield.db

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ src/
COPY data/policy_docs/ data/policy_docs/
COPY --from=builder /build/data/raw/login_events.csv /build/data/raw/graph_entities.csv data/raw/
COPY --from=builder /build/data/processed/transactions_with_graph.csv data/processed/
COPY --from=builder /build/models/xgboost_with_graph_pipeline.joblib models/

# Fail the build -- not the first request -- if an artifact is missing or
# unloadable: load the data store + model and score one transaction.
RUN python -c "from bankshield.investigation import data_access as d; \
s = d.get_store(); t = s.transactions['transaction_id'].iloc[0]; \
r = d.get_risk_score(t); assert 0.0 <= r['risk_score'] <= 1.0, r; \
print('artifact check ok:', len(s.transactions), 'transactions')"

# The only writable path the app needs (besides /tmp): mount a volume here
# so state survives container restarts and replacement.
RUN mkdir -p /app/var && chown bankshield:bankshield /app/var
VOLUME ["/app/var"]

USER bankshield:bankshield
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# One worker by default. Approvals, cases and users are durable and safe
# across processes (decisions are conditional updates), but A2A task state
# and the login throttle are per-process and SQLite allows a single writer;
# scale out with PostgreSQL behind BANKSHIELD_DATABASE_URL.
CMD ["uvicorn", "bankshield.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
