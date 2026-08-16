"""Phase 5: defensive AI security layer for the investigation agent.

This package does not change what the Phase 1-4 system does -- it adds
structural controls around it: input validation, fail-closed containment,
and retrieval-scoped citation checking. Nothing here is a keyword filter;
each control enforces a schema, an allowlist, or a provenance check that
holds regardless of how a malicious instruction is worded.

See `security/redteam_cases.yaml` and `reports/phase5_redteam_report.md`
for the red-team suite this layer is evaluated against.
"""

from __future__ import annotations
