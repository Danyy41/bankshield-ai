# Changelog

All notable changes to BankShield AI. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Ruff lint step in CI (`ruff==0.16.10`, rules `F` and `E9`, configured in
  `ruff.toml`); five unused imports removed (#30).

## [1.0.0] - 2026-10-21

First tagged version of the offline-first portfolio project. Each entry links
the PR or commits that delivered it, and the retrospective issue that
documents it.

### Added
- **ML fraud detection:** synthetic seeded data, a chronological split, a
  logistic-regression baseline and XGBoost (`ea1f44f`, #14); cyber-authentication
  telemetry features (#1, #15); graph-based mule-ring features (#2, #16).
  Final model: ROC-AUC 0.937, PR-AUC 0.572 at 1.38% fraud prevalence.
- **AI-assisted investigation:** a tool-use agent, policy RAG with
  retrieval-scoped citations, and a human-approval gate for case creation,
  served by a FastAPI service (#3, #17). Follow-ups added the LLM mode switch,
  a demo endpoint and a web frontend (`5931047`, `5ac2674`, `4b4ee1b`, #18).
- **AI security red team:** 24 synthetic attacks with defenses in validators
  and guardrails (`6c0eac7`, `5c83e28`, #19). The suite now has 30 cases,
  all blocked.
- **CI:** a GitHub Actions pipeline regenerating data and models, with
  `pytest` and a red-team gate (#4, #20).
- **Reproducibility:** pinned dependencies with a lock file, seeded IDs,
  recorded environments, and a CI check of regenerated metrics (#5, #21).
- **Docker:** a hardened multi-stage image (non-root, read-only filesystem,
  healthcheck) with a CI smoke test (#6, #22).
- **MCP server:** five read-only investigation tools over stdio, checked in
  CI by the MCP Inspector CLI (#7, #23).
- **A2A agents:** triage and investigation agents on the official A2A SDK;
  proposals still require a human reviewer (#8, #24).
- **Security and durable state:** JWT authentication, role-based access,
  four-eyes approval, transactional decisions, an append-only audit log, and
  state that survives restarts (#9, #25).
- **Optional Amazon Bedrock client:** off by default, with least-privilege
  IAM, fail-closed error handling and diagnostics (#10, #11, #26). Live
  inference was attempted but didn't succeed, so the integration is
  unverified and no live results are included.
- **Documentation:** a concise README, `docs/design.md` and
  `docs/bedrock.md` (#13, #27).
- **Development process:** `docs/development-process.md`, issue and PR
  templates, milestones, a project board, and this changelog (#37, #28).

### Verification at this version
- `pytest`: 315 passed
- Red team: 30/30 attacks blocked
- Reproducibility: 7/7 metrics files match
- Docker smoke test and MCP Inspector check: passed in CI

### Known limitations
Synthetic data only. The default LLM client is a deterministic offline
agent, so evaluation results measure the controls rather than a live model.
Authentication uses a local identity store (no IdP or MFA). CORS is open.
SQLite allows a single writer. See the [Backlog](https://github.com/Danyy41/bankshield-ai/milestone/9).

[Unreleased]: https://github.com/Danyy41/bankshield-ai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Danyy41/bankshield-ai/releases/tag/v1.0.0
