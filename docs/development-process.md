# Development process

How BankShield AI is planned, built, tested, reviewed, released and
maintained, and how it actually got here. Every artifact referenced below
exists in this repository or on its
[project board](https://github.com/users/Danyy41/projects/1).

## Lifecycle

| Stage | What happens | Where it lives |
|---|---|---|
| **Requirements** | Each change starts as a GitHub issue with a problem statement and measurable acceptance criteria | [Issues](https://github.com/Danyy41/bankshield-ai/issues), [issue templates](../.github/ISSUE_TEMPLATE/) |
| **Planning** | Issues are grouped into milestones and prioritized on the board (Backlog → In Progress → In Review → Done) | [Milestones](https://github.com/Danyy41/bankshield-ai/milestones), [project board](https://github.com/users/Danyy41/projects/1) |
| **Implementation** | Work happens on a branch, never directly on `main`, and is opened as a pull request that references its issue | [Pull requests](https://github.com/Danyy41/bankshield-ai/pulls?q=is%3Apr) |
| **Automated testing** | CI runs on every push and PR: it regenerates the data and models, then runs `pytest`, the red-team suite, the reproducibility check, the MCP Inspector check and the hardened Docker smoke test | [`tests.yml`](../.github/workflows/tests.yml), [`tests/`](../tests/), [`security/`](../security/) |
| **Code review** | The PR template records the linked issue and test evidence. A PR is merged only when CI is green and the owner approves | [PR template](../.github/pull_request_template.md) |
| **Release** | Versions follow semantic versioning. Each release is a tag plus notes that mirror the changelog | [`CHANGELOG.md`](../CHANGELOG.md), [Releases](https://github.com/Danyy41/bankshield-ai/releases) |
| **Maintenance** | CI guards against regressions on every change; pinned dependencies and the reproducibility check catch drift in data and metrics; new work enters the Backlog as issues | [Backlog milestone](https://github.com/Danyy41/bankshield-ai/milestone/9) |

## Workflow for new work

1. **Open an issue** from a template, with acceptance criteria, a label and a
   milestone. The board adds it to **Backlog** automatically.
2. **Start work:** move the item to **In Progress** and create a branch, for
   example `feature/31-cost-based-threshold`.
3. **Check locally** before pushing:
   ```bash
   pytest
   python scripts/run_redteam.py
   python scripts/check_reproducibility.py
   ```
4. **Open a PR** using the template, with `Closes #<issue>` in the
   description. Move the item to **In Review**.
5. **CI must pass.** A red build is fixed on the branch. Tests are never
   skipped or disabled to get green.
6. **Merge** after review. The linked issue closes automatically and the board
   moves it to **Done**.
7. **Release** when a milestone is complete: update `CHANGELOG.md`, tag
   `vX.Y.Z` on a green `main` commit, and publish release notes.

**Definition of done:** all acceptance criteria met; tests added or updated
for the change; CI green; documentation updated where behavior changed; no
claims in the docs beyond what was verified.

## How the project actually got here

This process was formalized on 2026-10-21. Earlier work wasn't planned with
issues, and the history below is stated as it happened.

| Period | What happened | Evidence |
|---|---|---|
| 2026-08-15 to 08-16 | Phases 1–5 built: the ML pipeline, investigation agent and red-team suite. Some work was committed straight to `main`; PRs #1–#3 were merged seconds after opening. **No CI and no review** at this stage | Commits `ea1f44f`, `6c0eac7` and others; PRs #1–#3 |
| 2026-10-18 to 10-19 | CI introduced in PR #4. **Every PR from #4 to #13** ran the full CI pipeline and was merged by the owner after CI passed. Reproducibility, Docker, MCP, A2A, security and documentation were added this way | PRs #4–#13 (PR #12 closed unmerged, superseded by #13) |
| 2026-10-21 | Milestones, the project board and templates created. Past work was documented as **retrospective issues** #14–#27, each labeled `retrospective` and linked to its real PRs or commits. Future work was filed as issues #28–#36 | [Retrospective issues](https://github.com/Danyy41/bankshield-ai/issues?q=label%3Aretrospective) |

### AI-assisted development

BankShield AI was built with an AI coding assistant (Claude Code). The commit
history shows this openly: most commits are authored by "Claude".

- **The assistant:** wrote most of the code, tests and documentation, and ran
  the local checks.
- **The owner:** set the requirements and constraints for each step (for
  example, offline by default, no paid cloud calls without approval, a human
  approval gate), checked results against CI and test output, and decided
  whether to merge every PR.

There has been no independent second code reviewer. The quality gates are the
automated ones above (tests, red team, reproducibility and container smoke
tests on every PR), plus the owner's merge decision.
