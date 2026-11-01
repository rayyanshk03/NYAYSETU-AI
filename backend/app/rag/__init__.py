"""RAG (Retrieval-Augmented Generation) package for NyayaSetu AI."""

from .embeddings import MultilingualEmbedder, embedder
from .vector_store import ChromaVectorStore
from .retrieval import LegalRetriever, retrieval_service, retrieve_legal_passages
from .bm25_retriever import BM25LegalRetriever, get_bm25_retriever
from .hybrid_retriever import HybridRetriever, hybrid_retrieval_service, get_hybrid_retriever
from .reranker import LegalReranker, reranker_service, get_reranker
from .citation_validator import CitationValidator, CitationValidationResult
from .audit import CorpusAuditor, CorpusAuditReport
from .indexer import LegalCorpusIndexer, IndexingSummary
from .evaluator import RetrievalEvaluator, BenchmarkQuery, EvaluationMetricSummary

__all__ = [
    "MultilingualEmbedder",
    "embedder",
    "ChromaVectorStore",
    "LegalRetriever",
    "retrieval_service",
    "retrieve_legal_passages",
    "BM25LegalRetriever",
    "get_bm25_retriever",
    "HybridRetriever",
    "hybrid_retrieval_service",
    "get_hybrid_retriever",
    "LegalReranker",
    "reranker_service",
    "get_reranker",
    "CitationValidator",
    "CitationValidationResult",
    "CorpusAuditor",
    "CorpusAuditReport",
    "LegalCorpusIndexer",
    "IndexingSummary",
    "RetrievalEvaluator",
    "BenchmarkQuery",
    "EvaluationMetricSummary",
]
