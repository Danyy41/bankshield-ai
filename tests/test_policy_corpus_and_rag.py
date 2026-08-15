"""Tests for Phase 4 policy-document chunking and RAG retrieval."""

from __future__ import annotations

from bankshield.investigation.policy_corpus import load_policy_chunks
from bankshield.investigation.rag import PolicyRetriever, get_retriever, search_policy


def test_load_policy_chunks_parses_all_documents():
    chunks = load_policy_chunks()
    assert len(chunks) > 0
    doc_ids = {c.doc_id for c in chunks}
    # The six synthetic policy documents this repo ships.
    assert doc_ids == {
        "POL-AML-001",
        "POL-FRAUD-002",
        "POL-ATO-003",
        "POL-CASE-004",
        "POL-GRAPH-006",
        "POL-KYC-005",
    }


def test_chunk_ids_are_unique():
    chunks = load_policy_chunks()
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_section_labels_do_not_include_stray_period():
    chunks = load_policy_chunks()
    # Regression guard: source docs mix "## §1. Title" (trailing period)
    # and "### §2.1 Title" (no period) -- the parser must normalize both to
    # a clean "§N" / "§N.M" label, not carry the period into the citation.
    for c in chunks:
        assert not c.section.endswith("."), c.section


def test_every_chunk_has_nonempty_text():
    for c in load_policy_chunks():
        assert c.text.strip()


def test_retriever_finds_relevant_ato_section_for_ato_query():
    retriever = PolicyRetriever(load_policy_chunks())
    results = retriever.search("failed login attempts new device account takeover", top_k=3)
    assert results
    top_doc_ids = {chunk.doc_id for chunk, _score in results}
    assert "POL-ATO-003" in top_doc_ids


def test_retriever_finds_relevant_mule_ring_section_for_graph_query():
    retriever = PolicyRetriever(load_policy_chunks())
    results = retriever.search("shared device shared IP mule ring network", top_k=3)
    assert results
    top_doc_ids = {chunk.doc_id for chunk, _score in results}
    assert "POL-GRAPH-006" in top_doc_ids or "POL-AML-001" in top_doc_ids


def test_get_by_id_roundtrips_with_search_results():
    retriever = get_retriever()
    hits = search_policy("consequential action human approval")
    assert hits
    first = hits[0]
    chunk = retriever.get_by_id(first["doc_id"], first["section"])
    assert chunk is not None
    assert chunk.text == first["text"]


def test_get_by_id_returns_none_for_unknown_section():
    retriever = get_retriever()
    assert retriever.get_by_id("POL-DOES-NOT-EXIST", "§1") is None


def test_search_policy_returns_citation_ready_dicts():
    results = search_policy("SAR filing criteria", top_k=2)
    for r in results:
        assert {"doc_id", "section", "title", "text", "score"} <= r.keys()
        assert r["doc_id"].startswith("POL-")
        assert r["section"].startswith("§")
