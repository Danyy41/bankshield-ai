"""Loads and chunks the synthetic policy documents in data/policy_docs/.

Chunking follows the documents' own section structure (`## SS1. ...`) rather
than a fixed-size sliding window: every synthetic policy doc in this repo is
written with one coherent idea per section, so section boundaries are also
the right citation granularity -- a citation like "POL-ATO-003 SS2.1" both
retrieves cleanly and reads naturally in a generated explanation. Sections
that run long are still split further (with overlap) so no chunk exceeds
`config.POLICY_CHUNK_SIZE_CHARS`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bankshield import config

_DOC_ID_RE = re.compile(r"^#\s+([A-Z0-9-]+)\s*[—-]\s*(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^(#{2,3})\s+§(\d+(?:\.\d+)?)\.?\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class PolicyChunk:
    doc_id: str
    doc_title: str
    section: str
    section_title: str
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id} {self.section}"


def _split_long_chunk(text: str, section_label: str) -> list[str]:
    size = config.POLICY_CHUNK_SIZE_CHARS
    overlap = config.POLICY_CHUNK_OVERLAP_CHARS
    if len(text) <= size:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return parts


def _parse_document(raw: str) -> tuple[str, str, list[PolicyChunk]]:
    doc_match = _DOC_ID_RE.search(raw)
    if not doc_match:
        raise ValueError("policy document is missing a '# DOC-ID — Title' header")
    doc_id, doc_title = doc_match.group(1), doc_match.group(2).strip()

    section_matches = list(_SECTION_RE.finditer(raw))
    chunks: list[PolicyChunk] = []
    for i, match in enumerate(section_matches):
        section_label = f"§{match.group(2)}"
        section_title = match.group(3).strip()
        body_start = match.end()
        body_end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(raw)
        body = raw[body_start:body_end].strip()
        if not body:
            # A parent heading (e.g. "## §2. Risk-scoring thresholds")
            # immediately followed by its subsections has no text of its
            # own to cite -- only its subsections are chunk-worthy.
            continue
        for j, part in enumerate(_split_long_chunk(body, section_label)):
            label = section_label if j == 0 else f"{section_label} (cont. {j})"
            chunks.append(
                PolicyChunk(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    section=label,
                    section_title=section_title,
                    text=part,
                )
            )
    return doc_id, doc_title, chunks


def load_policy_chunks_from(directory) -> list[PolicyChunk]:
    """Parse every .md file in an arbitrary directory into section-level
    chunks, using the same parsing rules as `load_policy_chunks`. Used by
    the Phase 5 red-team suite to build an isolated retriever over
    `security/malicious_policy_fixtures/` -- a directory deliberately kept
    separate from `config.POLICY_DOCS_DIR` -- without ever touching the
    production corpus. Raises FileNotFoundError if the directory is
    missing/empty, same as `load_policy_chunks`."""
    from pathlib import Path

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")

    doc_paths = sorted(directory.glob("*.md"))
    if not doc_paths:
        raise FileNotFoundError(f"no policy documents found in {directory}")

    all_chunks: list[PolicyChunk] = []
    for path in doc_paths:
        # Explicit encoding, not the platform default: these documents are
        # UTF-8 (they use "§" and em dashes). Path.read_text() with no
        # encoding= falls back to locale.getpreferredencoding(), which is
        # UTF-8 on Linux/Mac but frequently cp1252 on Windows -- under
        # cp1252 the "§" byte sequence is misdecoded, _SECTION_RE stops
        # matching entirely, and every document parses to zero chunks
        # (silently, no exception) rather than raising here.
        _, _, chunks = _parse_document(path.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)
    return all_chunks


def load_policy_chunks() -> list[PolicyChunk]:
    """Parse every .md file in config.POLICY_DOCS_DIR into section-level
    chunks. Raises FileNotFoundError if the directory is missing/empty."""
    return load_policy_chunks_from(config.POLICY_DOCS_DIR)
