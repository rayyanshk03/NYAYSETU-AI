"""Hybrid Retrieval Pipeline for NyayaSetu AI.

Combines dense semantic retrieval (ChromaDB cosine similarity) with sparse
lexical retrieval (BM25) using Reciprocal Rank Fusion (RRF). Supports:

  1. Dense retrieval  — ChromaDB vector similarity (multilingual embeddings)
  2. Sparse retrieval — BM25 keyword/section-number matching
  3. RRF fusion      — Combines both ranked lists without requiring score
                       calibration between the two systems
  4. Metadata filtering by Act name, document_id, or legal topic
  5. Deduplication   — ensures no passage appears twice in results
  6. Latency tracking per retrieval stage

Reciprocal Rank Fusion:
  For each document d retrieved by system s at rank r:
      rrf_score(d) = Σ_s  1 / (k + r(s, d))
  where k=60 is the standard smoothing constant (Cormack et al., 2009).
  RRF is robust to score-scale differences between dense and sparse systems.

References:
  Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
  Reciprocal rank fusion outperforms condorcet and individual rank learning
  methods. SIGIR 2009.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.schemas.chat import SupportedLanguage
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedPassage
from app.rag.vector_store import ChromaVectorStore
from app.rag.bm25_retriever import BM25LegalRetriever, get_bm25_retriever
from app.rag.reranker import LegalReranker, get_reranker

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from Cormack et al. (2009)
RRF_K = 60


class HybridRetriever:
    """
    Hybrid legal retriever: Dense (ChromaDB) + Sparse (BM25) → RRF fusion → Reranker.

    Usage:
        retriever = HybridRetriever()
        response = retriever.retrieve(query="grounds for divorce", top_k=5)
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_retriever: Optional[BM25LegalRetriever] = None,
        reranker: Optional[LegalReranker] = None,
        dense_candidates: int = 20,
        sparse_candidates: int = 20,
    ):
        """
        Args:
            vector_store:      ChromaDB vector store (uses singleton if None)
            bm25_retriever:    BM25 retriever (uses singleton if None)
            reranker:          Cross-encoder reranker (uses singleton if None)
            dense_candidates:  Number of candidates from dense retrieval before fusion
            sparse_candidates: Number of candidates from sparse retrieval before fusion
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25 = bm25_retriever or get_bm25_retriever()
        self.reranker = reranker or get_reranker()
        self.dense_candidates = dense_candidates
        self.sparse_candidates = sparse_candidates
        self._bm25_loaded: bool = False

    def _ensure_bm25_loaded(self) -> None:
        """
        Lazy-load the BM25 index from ChromaDB if not yet built.
        Fetches all documents from the collection to build the full BM25 index.
        """
        if self._bm25_loaded and self.bm25.is_built:
            return

        try:
            col = self.vector_store.get_collection()
            count = col.count()
            if count == 0:
                logger.info("BM25: corpus is empty, skipping BM25 index build.")
                self._bm25_loaded = True
                return

            logger.info("Building BM25 index from %d corpus documents…", count)
            t0 = time.perf_counter()

            # Fetch all documents in one call (ChromaDB supports this)
            result = col.get(include=["documents", "metadatas"])
            ids = result.get("ids", [])
            docs = result.get("documents", []) or []
            metas = result.get("metadatas", []) or []

            records = [
                {
                    "chunk_id": cid,
                    "legal_text": doc or "",
                    "metadata": meta or {},
                }
                for cid, doc, meta in zip(ids, docs, metas)
            ]
            self.bm25.build_index(records)
            self._bm25_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info("BM25 index built in %.2fs (%d docs).", elapsed, len(records))
        except Exception as e:
            logger.error("Failed to build BM25 index: %s", e)
            self._bm25_loaded = True  # Prevent repeated failures

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[SupportedLanguage] = None,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None,
        use_hybrid: bool = True,
        rerank: bool = True,
    ) -> RetrievalResponse:
        """
        Execute hybrid retrieval and return the top_k best passages.

        Steps:
          1. Check corpus size (return empty-corpus status if 0)
          2. Dense retrieval from ChromaDB (top dense_candidates)
          3. Sparse BM25 retrieval (top sparse_candidates) if use_hybrid=True
          4. RRF fusion of dense + sparse ranked lists
          5. Threshold filtering (min_score applied to dense similarity)
          6. Deduplication by chunk_id
          7. Second-stage cross-encoder reranking (if rerank=True)
          8. Return structured RetrievalResponse with latency metadata

        Args:
            query:       User legal query string
            top_k:       Max passages to return after fusion/reranking
            language:    Query language for logging
            document_id: Optional filter to restrict to one Act
            min_score:   Minimum dense similarity score (default: settings.MIN_SIMILARITY_SCORE)
            use_hybrid:  If False, fall back to dense-only retrieval
            rerank:      If True, run CrossEncoder reranking on candidate pool
        """
        clean_query = query.strip() if query else ""
        effective_lang = language or SupportedLanguage.ENGLISH
        threshold = min_score if min_score is not None else settings.MIN_SIMILARITY_SCORE

        if not clean_query:
            return RetrievalResponse(
                query="",
                language=effective_lang,
                total_results=0,
                insufficient_evidence=True,
                evidence_note="Query string cannot be empty.",
                passages=[],
            )

        # Check corpus
        corpus_size = self.vector_store.count()
        if corpus_size == 0:
            return RetrievalResponse(
                query=clean_query,
                language=effective_lang,
                total_results=0,
                insufficient_evidence=True,
                evidence_note=(
                    "The legal knowledge base currently contains 0 indexed provisions. "
                    "Run: python -m app.ingestion.pipeline && python -m app.rag.indexer"
                ),
                passages=[],
            )

        latency: Dict[str, float] = {}

        # ── 1. Dense retrieval ──────────────────────────────────────────────
        t0 = time.perf_counter()
        where_filter: Optional[Dict[str, Any]] = (
            {"document_id": document_id.strip()} if document_id else None
        )
        dense_results = self.vector_store.query(
            query_text=clean_query,
            top_k=min(self.dense_candidates, corpus_size),
            filter_metadata=where_filter,
        )
        latency["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Build dense rank map: chunk_id → rank (1-indexed)
        dense_rank: Dict[str, int] = {
            r["chunk_id"]: idx + 1 for idx, r in enumerate(dense_results)
        }
        # Map chunk_id → dense result record
        dense_map: Dict[str, Dict[str, Any]] = {r["chunk_id"]: r for r in dense_results}

        # ── 2. Sparse BM25 retrieval ────────────────────────────────────────
        sparse_rank: Dict[str, int] = {}
        if use_hybrid:
            t1 = time.perf_counter()
            try:
                self._ensure_bm25_loaded()
                sparse_results = self.bm25.search(
                    query=clean_query,
                    top_k=self.sparse_candidates,
                    document_id=document_id,
                )
                sparse_rank = {
                    r["chunk_id"]: idx + 1 for idx, r in enumerate(sparse_results)
                }
            except Exception as e:
                logger.warning("BM25 search failed, using dense-only: %s", e)
            latency["sparse_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        # ── 3. Reciprocal Rank Fusion ───────────────────────────────────────
        all_chunk_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
        rrf_scores: Dict[str, float] = {}
        for cid in all_chunk_ids:
            score = 0.0
            if cid in dense_rank:
                score += 1.0 / (RRF_K + dense_rank[cid])
            if cid in sparse_rank:
                score += 1.0 / (RRF_K + sparse_rank[cid])
            rrf_scores[cid] = score

        # Sort by RRF score descending
        ranked_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)

        # ── 4. Build passage objects ────────────────────────────────────────
        passages: List[RetrievedPassage] = []
        seen_chunk_ids: set = set()  # deduplication

        for cid in ranked_ids:
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)

            # Prefer dense result record (has similarity score); fall back to sparse
            if cid in dense_map:
                r = dense_map[cid]
            else:
                # Chunk only in BM25 results — fetch from ChromaDB by id
                try:
                    fetch = self.vector_store.get_collection().get(
                        ids=[cid],
                        include=["documents", "metadatas"],
                    )
                    if fetch["ids"]:
                        r = {
                            "chunk_id": cid,
                            "legal_text": (fetch["documents"] or [""])[0],
                            "metadata": (fetch["metadatas"] or [{}])[0],
                            "score": 0.0,  # No dense score available
                        }
                    else:
                        continue
                except Exception:
                    continue

            meta = r.get("metadata", {})
            dense_similarity = r.get("score", 0.0)

            # Apply minimum dense similarity filter only to chunks that appeared
            # in the dense results. BM25-only results pass through.
            if cid in dense_rank and dense_similarity < threshold:
                continue

            passages.append(
                RetrievedPassage(
                    chunk_id=cid,
                    document_id=meta.get("document_id", "unknown"),
                    document_title=meta.get("document_title", "Unknown Statute"),
                    act_name=meta.get("act_name") or None,
                    section_number=meta.get("section_number") or None,
                    section_title=meta.get("section_title") or None,
                    subsection_id=meta.get("subsection_id") or None,
                    page_start=int(meta.get("page_start", 1)),
                    page_end=int(meta.get("page_end", 1)),
                    page_number=int(meta.get("page_start", 1)),
                    source_url=meta.get("source_url", "https://www.indiacode.nic.in/"),
                    publication_date=meta.get("publication_date") or None,
                    effective_date=meta.get("effective_date") or None,
                    document_version=meta.get("document_version") or None,
                    legal_text=r.get("legal_text", ""),
                    score=round(dense_similarity, 4),
                    is_split_fallback=bool(meta.get("is_split_fallback", False)),
                    metadata={
                        **meta,
                        "rrf_score": round(rrf_scores.get(cid, 0.0), 6),
                        "dense_rank": dense_rank.get(cid),
                        "sparse_rank": sparse_rank.get(cid),
                        "retrieval_latency_ms": latency,
                    },
                )
            )

            candidate_pool_limit = max(top_k * 3, 15) if rerank else top_k
            if len(passages) >= candidate_pool_limit:
                break

        # ── 5. Cross-Encoder Reranking ──────────────────────────────────────
        if rerank and passages:
            t_rerank = time.perf_counter()
            passages = self.reranker.rerank(
                query=clean_query,
                passages=passages,
                top_k=top_k,
            )
            latency["rerank_ms"] = round((time.perf_counter() - t_rerank) * 1000, 1)
            for p in passages:
                if p.metadata:
                    p.metadata["retrieval_latency_ms"] = latency
        else:
            passages = passages[:top_k]

        insufficient = not passages
        evidence_note: Optional[str] = None
        if insufficient:
            evidence_note = (
                f"No statutory provisions met the retrieval threshold of {threshold:.2f} "
                f"(dense similarity). Hybrid retrieval with BM25 also found no matching passages. "
                "Returning insufficient evidence rather than guessing."
            )

        from datetime import datetime, timezone
        return RetrievalResponse(
            status="success",
            query=clean_query,
            language=effective_lang,
            total_results=len(passages),
            insufficient_evidence=insufficient,
            evidence_note=evidence_note,
            passages=passages,
            timestamp=datetime.now(timezone.utc),
        )

    def retrieve_with_provenance(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[SupportedLanguage] = None,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None,
        use_hybrid: bool = True,
        rerank: bool = True,
    ) -> RetrievalResponse:
        """
        Alias to retrieve() that matches the orchestrator's expected method name.
        Returns RetrievalResponse with .passages and .insufficient_evidence.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            language=language,
            document_id=document_id,
            min_score=min_score,
            use_hybrid=use_hybrid,
            rerank=rerank,
        )


# ── Convenience singleton ───────────────────────────────────────────────────

_hybrid_retriever_instance: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Return or create the module-level HybridRetriever singleton."""
    global _hybrid_retriever_instance
    if _hybrid_retriever_instance is None:
        _hybrid_retriever_instance = HybridRetriever()
    return _hybrid_retriever_instance


hybrid_retrieval_service = get_hybrid_retriever()
