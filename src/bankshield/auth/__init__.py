"""Authentication and role-based authorization for BankShield.

- passwords.py -- scrypt password hashing (stdlib, memory-hard)
- users.py     -- durable user accounts with one role each
- tokens.py    -- short-lived signed JWT access tokens
- context.py   -- the authenticated principal for the current request
- dependencies.py -- FastAPI guards: authenticate, then authorize by role
- cli.py       -- account provisioning (there is deliberately no HTTP API
                  for creating users or changing roles)

Roles:
    analyst   read data, run investigations, view approvals and cases
    reviewer  everything an analyst can read, plus decide approvals
    admin     read everything, including the audit log; cannot decide
    agent     service accounts for A2A callers: read data, call the agents
"""
