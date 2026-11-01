"""Legal retrieval service for NyayaSetu AI.

Accepts user legal queries in English, Hindi, or Kannada, searches the ChromaDB
vector index, applies relevance thresholds, and formats authoritative statutory passages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.core.config import settings
from app.schemas.chat import SupportedLanguage
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedPassage
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class LegalRetriever:
    """Service providing semantic retrieval over authoritative Indian family law statutes."""

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[SupportedLanguage] = None,
        document_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalResponse:
        """
        Execute semantic retrieval for a user legal query.

        Args:
            query: The legal query text.
            top_k: Maximum number of passages to return (default: 5).
            language: Target query language (en, hi, kn).
            document_id: Optional filter for a specific Act (e.g. 'hma_1955').
            min_score: Minimum similarity score threshold.
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

        # Check if vector store is empty
        corpus_size = self.vector_store.count()
        if corpus_size == 0:
            return RetrievalResponse(
                query=clean_query,
                language=effective_lang,
                total_results=0,
                insufficient_evidence=True,
                evidence_note=(
                    "The legal knowledge base currently contains 0 indexed provisions. "
                    "Please download official Bare Acts from India Code into data/raw/, "
                    "run ingestion (python -m app.ingestion.pipeline), and index the corpus "
                    "(python -m app.rag.indexer)."
                ),
                passages=[],
            )

        # Apply metadata filter if requested
        where_filter: Optional[Dict[str, Any]] = None
        if document_id:
            where_filter = {"document_id": document_id.strip()}

        # Query ChromaDB
        raw_results = self.vector_store.query(
            query_text=clean_query,
            top_k=top_k,
            filter_metadata=where_filter,
        )

        passages: List[RetrievedPassage] = []
        for r in raw_results:
            meta = r.get("metadata", {})
            score = r.get("score", 0.0)

            # Map raw metadata back to verified fields
            passages.append(
                RetrievedPassage(
                    chunk_id=r["chunk_id"],
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
                    legal_text=r["legal_text"],
                    score=score,
                    is_split_fallback=bool(meta.get("is_split_fallback", False)),
                    metadata=meta,
                )
            )

        # Filter by minimum relevance threshold
        qualified_passages = [p for p in passages if p.score >= threshold]

        insufficient = False
        evidence_note = None

        if not qualified_passages:
            insufficient = True
            if passages:
                top_score = passages[0].score
                evidence_note = (
                    f"No statutory provisions met the relevance threshold of {threshold:.2f} "
                    f"(highest retrieved similarity score was {top_score:.2f}). "
                    f"Returning insufficient evidence rather than guessing."
                )
            else:
                evidence_note = "No matching legal passages were retrieved for the query."

        return RetrievalResponse(
            status="success",
            query=clean_query,
            language=effective_lang,
            total_results=len(qualified_passages),
            insufficient_evidence=insufficient,
            evidence_note=evidence_note,
            passages=qualified_passages,
            timestamp=datetime.now(timezone.utc),
        )


retrieval_service = LegalRetriever()


def retrieve_legal_passages(
    query: str,
    top_k: int = 5,
    language: Optional[SupportedLanguage] = None,
    document_id: Optional[str] = None,
    min_score: Optional[float] = None,
    vector_store: Optional[ChromaVectorStore] = None,
) -> RetrievalResponse:
    """
    Reusable retrieval function as required by Step 3 specification.

    Searches the authoritative legal knowledge base and returns structured
    passages with:
    - Chunk ID (`chunk_id`)
    - Retrieved legal text (`legal_text`)
    - Document title (`document_title`)
    - Act name (`act_name`)
    - Section number (`section_number`, if available)
    - Page number (`page_number`, if available)
    - Original source URL (`source_url`, if available)
    - Relevance score (`score`, cosine similarity; not a probability of legal validity)
    - Available metadata dictionary (`metadata`)

    Args:
        query: User's legal question string.
        top_k: Number of top results to return (default: 5).
        language: SupportedLanguage (en, hi, kn).
        document_id: Optional filter for a specific Act (e.g. 'hma_1955').
        min_score: Optional minimum relevance score cutoff.
        vector_store: Optional vector store override (useful for test fixtures).
    """
    retriever = LegalRetriever(vector_store=vector_store) if vector_store else retrieval_service
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        language=language,
        document_id=document_id,
        min_score=min_score,
    )
