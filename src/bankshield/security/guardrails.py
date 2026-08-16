"""Structural containment and provenance controls for the agent loop.

Three controls live here, all enforced in code rather than relied upon
from a prompt instruction:

1. **Fail-closed tool execution** (`safe_execute_tool`): a tool call for an
   unknown/disallowed name raises `tools.ToolExecutionError` by design
   (see tools.py) -- that contract is intentionally left alone since
   tests depend on it. What was missing is a caller that contains that
   exception instead of letting it crash the whole investigation. This
   wrapper is that containment: an unauthorized tool request becomes an
   error tool-result the loop can continue past, not an unhandled 500.

2. **Retrieval-scoped citation filtering** (`filter_citations_to_retrieved`):
   a citation is only trustworthy if it was actually surfaced by this
   run's retrieval (the automatic payload search, or an explicit
   `search_policy` call) -- not merely because a chunk with that
   doc_id/section happens to exist somewhere in the corpus. This closes
   the gap where a narrative could cite a real-but-never-retrieved policy
   section and have it pass as "verified" just because the ID happened to
   resolve. Retrieved documents are treated as untrusted evidence: their
   *content* never has authority to change agent behavior, and their mere
   *existence* never has authority to back a citation the agent didn't
   actually look up this run.

3. **Sanitized error responses** (`sanitize_error_response`): used by the
   API layer's catch-all exception handler so an unexpected internal
   error (stack trace, file paths, internal exception message) is never
   echoed back to a caller -- fail closed with a generic message instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bankshield.investigation import tools
from bankshield.investigation.schemas import ToolCallRecord

if TYPE_CHECKING:
    from bankshield.investigation.rag import PolicyRetriever
    from bankshield.investigation.schemas import InvestigationPayload, PolicyCitation

logger = logging.getLogger("bankshield.security")


def safe_execute_tool(name: str, tool_input: dict) -> tools.ToolExecutionResult:
    """Execute a tool call, containing an unauthorized/unknown tool name as
    a fail-closed error result instead of propagating the exception.

    `tools.execute_tool` itself still raises `ToolExecutionError` for an
    unknown name (its existing, tested contract) -- this wrapper is the
    agent loop's boundary, not a change to that contract.
    """
    try:
        return tools.execute_tool(name, tool_input)
    except tools.ToolExecutionError as exc:
        logger.warning("blocked unauthorized tool request: %s", exc)
        message = str(exc)
        return tools.ToolExecutionResult(
            output={"error": message},
            record=ToolCallRecord(
                tool_name=name,
                input=tool_input if isinstance(tool_input, dict) else {},
                output={"error": message},
                output_summary=f"error: {message}",
                is_error=True,
                latency_ms=0.0,
            ),
        )


def retrieved_chunk_ids(payload: "InvestigationPayload", tool_call_records: list[ToolCallRecord]) -> set[str]:
    """Every policy chunk actually surfaced during this run: the automatic
    payload-building search plus any explicit `search_policy` tool calls.
    A citation not in this set was not retrieved this run, regardless of
    whether it resolves to a real chunk elsewhere in the corpus."""
    ids = {f"{c.doc_id} {c.section}" for c in payload.retrieved_policy}
    for call in tool_call_records:
        if call.tool_name == "search_policy" and not call.is_error:
            for hit in call.output.get("results", []):
                doc_id = hit.get("doc_id")
                section = hit.get("section")
                if doc_id and section:
                    ids.add(f"{doc_id} {section}")
    return ids


def filter_citations_to_retrieved(
    narrative: str,
    citation_pattern,
    retriever: "PolicyRetriever",
    retrieved_ids: set[str],
) -> list["PolicyCitation"]:
    """Extract `[DOC-ID §section]`-shaped citations from `narrative` and
    keep only those that both (a) resolve to a real chunk in the corpus
    and (b) were actually retrieved during this run (`retrieved_ids`).
    A citation satisfying only (a) is exactly the fabricated-grounding
    failure mode this exists to catch: a real policy ID/section quoted
    from the model's training data or an injected instruction, never
    actually looked up."""
    from bankshield.investigation.schemas import PolicyCitation

    citations: list[PolicyCitation] = []
    seen: set[tuple[str, str]] = set()
    for doc_id, section in citation_pattern.findall(narrative):
        key = (doc_id, section)
        if key in seen:
            continue
        seen.add(key)

        if f"{doc_id} {section}" not in retrieved_ids:
            continue

        chunk = retriever.get_by_id(doc_id, section)
        if chunk is not None:
            citations.append(
                PolicyCitation(
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    title=f"{chunk.doc_title} — {chunk.section_title}",
                    text=chunk.text,
                    score=1.0,
                )
            )
    return citations


_GENERIC_ERROR_MESSAGE = "internal error processing request"


def sanitize_error_response(exc: BaseException) -> dict:
    """Fail-closed shape for an unexpected exception: log full detail
    server-side, return only a generic message to the caller. Never
    echoes the exception's own message (which could contain file paths,
    internal identifiers, or library internals) back over the API."""
    logger.exception("unhandled exception while serving request: %s", type(exc).__name__)
    return {"detail": _GENERIC_ERROR_MESSAGE}
