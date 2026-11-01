"""BM25 lexical retriever for Indian family law legal passages.

Provides exact-term matching for Act names, section numbers, statutory
keywords, and other precise legal references that semantic search may
miss due to vocabulary mismatch. Used as the sparse leg of hybrid
Reciprocal Rank Fusion retrieval.

Design notes:
- Corpus is loaded from ChromaDB at initialization to stay consistent
  with the vector index.
- BM25 index is rebuilt on demand when corpus changes (reset_index).
- Legal tokenization preserves Act/section tokens, numbers, and Indic
  Unicode characters; it does NOT normalise legal references.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _tokenize_legal(text: str) -> List[str]:
    """
    Tokenize a legal text string.

    Preserves:
    - Section/sub-section identifiers (e.g., "13B", "125(1)")
    - Act year numbers (e.g., "1955", "2005")
    - Indic Unicode words (Devanagari, Kannada scripts)
    - Ordinary English words

    Removes only punctuation that carries no discriminative legal meaning.
    Does NOT lowercase so that "Act" is distinct from "act" only when intentional.
    Here we do lowercase for BM25 recall purposes.
    """
    if not text:
        return []

    # Normalise whitespace
    text = text.strip()
    text_lower = text.lower()

    # Preserve legal section identifiers (e.g. 13b, 125(1))
    section_tokens = re.findall(r'\b\d+[a-z]{0,3}\b|\b\d+\(\d+\)', text_lower)

    # General word tokenization: keep alphanumeric + Indic characters
    general_tokens = re.findall(
        r'[a-z0-9]+|[\u0900-\u097F]+|[\u0C80-\u0CFF]+',
        text_lower,
    )

    # Merge, preserving section tokens first (they are already in general_tokens)
    all_tokens = general_tokens
    # Deduplicate while preserving order
    seen: set = set()
    result = []
    for t in all_tokens:
        if t not in seen and len(t) >= 2:
            seen.add(t)
            result.append(t)

    return result if result else ["_empty_"]


class BM25LegalRetriever:
    """
    Sparse BM25 retriever over the legal document corpus.

    The corpus is a list of dicts with at minimum:
      - 'chunk_id': str
      - 'legal_text': str
      - 'metadata': dict

    Initialized lazily or explicitly via build_index().
    """

    def __init__(self) -> None:
        self._corpus_records: List[Dict[str, Any]] = []
        self._bm25 = None
        self._built: bool = False

    def build_index(self, corpus_records: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from a list of corpus records.

        Each record must have: chunk_id (str), legal_text (str), metadata (dict).
        """
        if not corpus_records:
            logger.warning("BM25: Empty corpus passed to build_index — index not built.")
            self._corpus_records = []
            self._bm25 = None
            self._built = False
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
            return

        self._corpus_records = corpus_records
        tokenized = [_tokenize_legal(r.get("legal_text", "")) for r in corpus_records]
        self._bm25 = BM25Okapi(tokenized)
        self._built = True
        logger.info("BM25 index built with %d documents.", len(corpus_records))

    def reset_index(self) -> None:
        """Clear the BM25 index."""
        self._corpus_records = []
        self._bm25 = None
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built and self._bm25 is not None

    @property
    def corpus_size(self) -> int:
        return len(self._corpus_records)

    def search(
        self,
        query: str,
        top_k: int = 20,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 search and return up to top_k ranked results.

        Returns list of dicts:
          - chunk_id: str
          - legal_text: str
          - metadata: dict
          - bm25_score: float  (raw BM25 score, NOT normalised to [0,1])
          - bm25_rank: int     (1-indexed rank in BM25 results)
        """
        if not self.is_built:
            return []

        query_tokens = _tokenize_legal(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Apply optional document filter
        if document_id:
            candidate_indices = [
                i for i, r in enumerate(self._corpus_records)
                if r.get("metadata", {}).get("document_id") == document_id
            ]
        else:
            candidate_indices = list(range(len(self._corpus_records)))

        # Sort by BM25 score descending
        ranked = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)

        results = []
        for rank, idx in enumerate(ranked[:top_k], start=1):
            record = self._corpus_records[idx]
            results.append({
                "chunk_id": record["chunk_id"],
                "legal_text": record.get("legal_text", ""),
                "metadata": record.get("metadata", {}),
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank,
            })

        return results


# Module-level singleton — populated by HybridRetriever on startup
_global_bm25_retriever: Optional[BM25LegalRetriever] = None


def get_bm25_retriever() -> BM25LegalRetriever:
    """Return the module-level BM25 retriever instance."""
    global _global_bm25_retriever
    if _global_bm25_retriever is None:
        _global_bm25_retriever = BM25LegalRetriever()
    return _global_bm25_retriever
