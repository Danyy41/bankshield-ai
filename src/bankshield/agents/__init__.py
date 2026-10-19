"""A2A multi-agent layer: a triage agent and an investigation agent that talk
to each other over the Agent2Agent protocol (official `a2a-sdk`).

- `investigator.py` -- wraps the existing investigation loop
  (`investigation.agent.run_investigation`) as an A2A agent.
- `triage.py` -- scores a batch of transactions with the read-only tools,
  and escalates Tier 1/2 alerts (POL-AML-001 §2.1) to the investigation
  agent over A2A.
- `server.py` -- mounts both agents' JSON-RPC endpoints and agent cards on
  the existing FastAPI app, so the investigation agent shares the API
  process's approval and case stores: a case proposal it files lands in
  the same queue a human reviewer decides through
  `POST /approvals/{id}/decision`.

Neither agent can create or approve a case: case creation stays behind the
existing human-approval gate (approvals.py, POL-CASE-004 §3).
"""
