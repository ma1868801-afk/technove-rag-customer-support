"""
Re-ranking stage.

Dense retrieval (bi-encoder embeddings like all-MiniLM-L6-v2) is fast but
scores query and document independently, which loses some precision. A
cross-encoder reads the (query, document) pair *together* and produces a
much more accurate relevance score. We use it as a second-stage filter:
retrieve a wide candidate pool cheaply with the vector store, then re-rank
that pool with the cross-encoder and keep only the best few for the LLM.
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src import config


class Reranker:
    """Thin wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME):
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = config.FINAL_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """Score every (query, doc) pair and return the top_k, sorted best-first."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
