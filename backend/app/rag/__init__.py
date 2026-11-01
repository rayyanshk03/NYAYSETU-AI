"""RAG (Retrieval-Augmented Generation) package for NyayaSetu AI."""

from .embeddings import MultilingualEmbedder, embedder
from .vector_store import ChromaVectorStore
from .retrieval import LegalRetriever, retrieval_service, retrieve_legal_passages
from .indexer import LegalCorpusIndexer, IndexingSummary
from .evaluator import RetrievalEvaluator, BenchmarkQuery, EvaluationMetricSummary

__all__ = [
    "MultilingualEmbedder",
    "embedder",
    "ChromaVectorStore",
    "LegalRetriever",
    "retrieval_service",
    "retrieve_legal_passages",
    "LegalCorpusIndexer",
    "IndexingSummary",
    "RetrievalEvaluator",
    "BenchmarkQuery",
    "EvaluationMetricSummary",
]
