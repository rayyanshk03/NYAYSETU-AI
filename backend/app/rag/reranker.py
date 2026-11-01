"""Legal cross-encoder reranker for NyayaSetu AI.

Implements a second-stage cross-encoder reranker on top of hybrid retrieval (dense + BM25).
While bi-encoders (dense embeddings) encode query and document independently,
cross-encoders process query-document pairs simultaneously with full cross-attention,
yielding significantly higher precision for legal text ranking.

Default model:
    cross-encoder/ms-marco-MiniLM-L-6-v2 (~25MB, runs entirely locally with zero API cost)

Fallback behavior:
    If the model fails to load or inference encounters an exception, the reranker
    logs a warning and returns the original passage ranking gracefully.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from app.core.config import settings
from app.schemas.retrieval import RetrievedPassage

logger = logging.getLogger(__name__)


class LegalReranker:
    """
    Second-stage legal cross-encoder reranker.

    Takes candidate passages from hybrid retrieval (dense + sparse RRF)
    and scores query-passage pairs jointly using full cross-attention.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.enabled = enabled if enabled is not None else settings.ENABLE_RERANKER
        self._model = None
        self._load_failed = False

    def _get_model(self):
        """Lazy-load the CrossEncoder model."""
        if not self.enabled or self._load_failed:
            return None

        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading CrossEncoder reranker model: %s", self.model_name)
                t0 = time.perf_counter()
                self._model = CrossEncoder(self.model_name)
                logger.info(
                    "CrossEncoder reranker loaded successfully in %.2fs",
                    time.perf_counter() - t0,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load CrossEncoder reranker '%s': %s. "
                    "Falling back to original hybrid rank.",
                    self.model_name,
                    e,
                )
                self._load_failed = True
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        passages: List[RetrievedPassage],
        top_k: Optional[int] = None,
    ) -> List[RetrievedPassage]:
        """
        Rerank a list of retrieved passages against the query using cross-attention.

        Args:
            query: The user's query string.
            passages: Candidate passages retrieved from Stage 1 (Hybrid RAG).
            top_k: Maximum number of reranked passages to return (defaults to settings.RERANKER_TOP_K or len(passages)).

        Returns:
            List of RetrievedPassage sorted by reranker score descending.
        """
        if not passages:
            return []

        target_top_k = top_k or settings.RERANKER_TOP_K
        if not self.enabled or not query.strip():
            return passages[:target_top_k]

        model = self._get_model()
        if model is None:
            # Graceful fallback: return original order
            return passages[:target_top_k]

        t0 = time.perf_counter()
        try:
            # Build query-document pairs
            # Truncate legal text slightly if exceptionally long to keep cross-attention fast
            pairs = [(query, p.legal_text[:1024]) for p in passages]
            raw_scores = model.predict(pairs)

            rerank_ms = round((time.perf_counter() - t0) * 1000, 1)

            # Pair each passage with its score
            scored_passages = []
            for passage, score in zip(passages, raw_scores):
                float_score = float(score)
                # Clone or update metadata
                meta = dict(passage.metadata or {})
                meta["original_dense_score"] = passage.score
                meta["reranker_score"] = round(float_score, 4)
                meta["reranker_latency_ms"] = rerank_ms
                meta["reranker_model"] = self.model_name

                # Update passage object
                passage.metadata = meta
                passage.score = round(float_score, 4)
                scored_passages.append((float_score, passage))

            # Sort descending by reranker score
            scored_passages.sort(key=lambda x: x[0], reverse=True)
            reranked = [p for _, p in scored_passages]

            logger.debug(
                "Reranked %d passages in %.1fms (top_k=%d)",
                len(passages),
                rerank_ms,
                target_top_k,
            )
            return reranked[:target_top_k]

        except Exception as e:
            logger.warning("Reranking failed (%s). Returning original passage ranking.", e)
            return passages[:target_top_k]


# Module singleton
_reranker_instance: Optional[LegalReranker] = None


def get_reranker() -> LegalReranker:
    """Return or initialize the LegalReranker singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LegalReranker()
    return _reranker_instance


reranker_service = get_reranker()
