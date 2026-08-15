"""RAG retrieval over the synthetic policy corpus.

Design choice: retrieval uses a local TF-IDF vector store (scikit-learn,
already a project dependency) rather than an embedding API. This keeps
retrieval deterministic and network-free, which matters for two things in
this repo specifically: the test suite (must run in CI/offline, same as
Phase 1-3's tests) and the evaluation harness (citation correctness has to
be reproducible run to run). The `PolicyRetriever` interface is small
enough to swap for a Bedrock Titan Embeddings + vector-DB backend later
without touching callers -- see the README's Phase 4 section for the
tradeoff this defers.
"""

from __future__ import annotations

from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bankshield import config
from bankshield.investigation.policy_corpus import PolicyChunk, load_policy_chunks


class PolicyRetriever:
    def __init__(self, chunks: list[PolicyChunk]):
        if not chunks:
            raise ValueError("PolicyRetriever requires at least one chunk")
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [f"{c.doc_title} {c.section_title} {c.text}" for c in chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = config.POLICY_SEARCH_TOP_K) -> list[tuple[PolicyChunk, float]]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]

    def get_by_id(self, doc_id: str, section: str) -> PolicyChunk | None:
        for chunk in self.chunks:
            if chunk.doc_id == doc_id and chunk.section == section:
                return chunk
        return None

    def all_chunk_ids(self) -> set[str]:
        return {c.chunk_id for c in self.chunks}


@lru_cache(maxsize=1)
def get_retriever() -> PolicyRetriever:
    return PolicyRetriever(load_policy_chunks())


def search_policy(query: str, top_k: int = config.POLICY_SEARCH_TOP_K) -> list[dict]:
    """Tool-facing entry point: returns citation-ready dicts."""
    retriever = get_retriever()
    results = retriever.search(query, top_k=top_k)
    return [
        {
            "doc_id": chunk.doc_id,
            "section": chunk.section,
            "title": f"{chunk.doc_title} — {chunk.section_title}",
            "text": chunk.text,
            "score": score,
        }
        for chunk, score in results
    ]
