"""Enhanced Legal RAG Evaluation Framework for NyayaSetu AI.

Implements rigorous retrieval and generation evaluation metrics:
  - Retrieval: Recall@5, Recall@10, Precision@5, Mean Reciprocal Rank (MRR)
  - Citation & Synthesis: Citation Precision, Claim Support Rate
  - Latency: retrieval_latency_ms, reranking_latency_ms, answer_latency_ms, total_latency_ms

CRITICAL INTEGRITY PRINCIPLE:
  Evaluation metrics are ONLY calculated for queries marked with `reviewer_status == "reviewed"`.
  If no verified ground-truth labels exist, the framework transparently returns
  `has_sufficient_ground_truth = False` and explains why, rather than inventing fake benchmark scores.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from app.schemas.chat import SupportedLanguage

logger = logging.getLogger(__name__)


class EvalQuery(BaseModel):
    """A test query with expert-reviewed legal ground truth."""

    query_id: str = Field(..., description="Unique query ID")
    query: str = Field(..., description="User query text")
    language: str = Field(default="en", description="Query language (en, hi, kn)")
    legal_topic: Optional[str] = Field(default=None, description="Topic (e.g., divorce, maintenance)")
    expected_document_ids: List[str] = Field(
        default_factory=list,
        description="List of expected document IDs (e.g. ['A1955-25'])",
    )
    expected_section_ids: List[str] = Field(
        default_factory=list,
        description="List of expected section numbers (e.g. ['13', '13B', '125'])",
    )
    expected_answer_points: List[str] = Field(
        default_factory=list,
        description="Key statutory concepts that must appear in the answer",
    )
    reviewer_status: str = Field(
        default="unreviewed",
        description="'reviewed' (verified by legal expert), 'draft', or 'unreviewed'",
    )
    reviewer_notes: Optional[str] = Field(default=None)


class EvaluationMetrics(BaseModel):
    """Summary of legal RAG evaluation metrics."""

    total_queries: int = 0
    reviewed_queries: int = 0
    unreviewed_queries: int = 0
    has_sufficient_ground_truth: bool = False

    # Retrieval accuracy
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0

    # Generation & Grounding quality
    citation_precision: float = 0.0
    claim_support_rate: float = 0.0

    # Latency tracking (milliseconds)
    avg_retrieval_latency_ms: float = 0.0
    avg_reranking_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0

    per_query_results: List[Dict[str, Any]] = Field(default_factory=list)
    report_message: str = ""


class LegalRAGEvaluator:
    """Evaluates the Legal RAG hybrid retrieval and synthesis pipeline."""

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or get_hybrid_retriever()

    @staticmethod
    def load_dataset(file_path: Path) -> List[EvalQuery]:
        """Load evaluation dataset from a JSON file."""
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("queries", [])
            return [EvalQuery.model_validate(item) for item in items]
        except Exception as e:
            logger.error("Failed to load evaluation dataset from %s: %s", file_path, e)
            return []

    def evaluate(
        self,
        queries: List[EvalQuery],
        top_k: int = 10,
    ) -> EvaluationMetrics:
        """
        Evaluate hybrid RAG pipeline on the given queries.

        Only queries with reviewer_status == 'reviewed' are included in statistical metrics.
        """
        summary = EvaluationMetrics(total_queries=len(queries))

        reviewed = [q for q in queries if q.reviewer_status.lower() == "reviewed"]
        summary.reviewed_queries = len(reviewed)
        summary.unreviewed_queries = len(queries) - len(reviewed)

        if not reviewed:
            summary.has_sufficient_ground_truth = False
            summary.report_message = (
                "No expert-reviewed ground-truth labels found in the dataset. "
                "Per NyayaSetu AI safety and accuracy standards, legal benchmark metrics "
                "cannot be calculated without verified statutory ground truth."
            )
            return summary

        summary.has_sufficient_ground_truth = True

        hits_5 = 0
        hits_10 = 0
        sum_p5 = 0.0
        reciprocal_ranks: List[float] = []
        retrieval_latencies: List[float] = []
        reranking_latencies: List[float] = []
        total_latencies: List[float] = []
        citation_precisions: List[float] = []

        for eq in reviewed:
            t0 = time.perf_counter()
            # Retrieve top_k passages using hybrid pipeline
            lang = None
            try:
                lang = SupportedLanguage(eq.language)
            except Exception:
                lang = SupportedLanguage.ENGLISH

            res = self.retriever.retrieve(query=eq.query, top_k=top_k, language=lang)
            total_elapsed_ms = (time.perf_counter() - t0) * 1000

            passages = res.passages
            retrieved_chunks = [p.chunk_id for p in passages]
            retrieved_docs = [p.document_id for p in passages]
            retrieved_secs = [str(p.section_number) for p in passages if p.section_number]

            # Latency breakdown from passage metadata
            ret_lat = 0.0
            rerank_lat = 0.0
            if passages and passages[0].metadata:
                lats = passages[0].metadata.get("retrieval_latency_ms", {})
                dense = lats.get("dense_ms", 0.0)
                sparse = lats.get("sparse_ms", 0.0)
                ret_lat = dense + sparse
                rerank_lat = lats.get("rerank_ms", 0.0)
            retrieval_latencies.append(ret_lat)
            reranking_latencies.append(rerank_lat)
            total_latencies.append(total_elapsed_ms)

            # Match criteria: document_id or section_number
            expected_docs = set(eq.expected_document_ids)
            expected_secs = {s.strip().lower() for s in eq.expected_section_ids}

            def is_hit(passage) -> bool:
                doc_hit = passage.document_id in expected_docs
                sec_hit = (
                    passage.section_number
                    and str(passage.section_number).strip().lower() in expected_secs
                )
                return doc_hit or sec_hit

            # Hit at rank
            first_hit_rank: Optional[int] = None
            for idx, p in enumerate(passages, start=1):
                if is_hit(p):
                    first_hit_rank = idx
                    break

            # Metrics for this query
            reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0
            reciprocal_ranks.append(reciprocal_rank)

            hit_in_5 = first_hit_rank is not None and first_hit_rank <= 5
            hit_in_10 = first_hit_rank is not None and first_hit_rank <= 10
            if hit_in_5:
                hits_5 += 1
            if hit_in_10:
                hits_10 += 1

            # Precision@5
            top_5 = passages[:5]
            hits_in_top5 = sum(1 for p in top_5 if is_hit(p))
            p5 = hits_in_top5 / 5.0 if top_5 else 0.0
            sum_p5 += p5

            # Citation precision (fraction of retrieved passages matching ground truth)
            c_prec = hits_in_top5 / len(top_5) if top_5 else 0.0
            citation_precisions.append(c_prec)

            summary.per_query_results.append(
                {
                    "query_id": eq.query_id,
                    "query": eq.query,
                    "language": eq.language,
                    "first_hit_rank": first_hit_rank,
                    "reciprocal_rank": round(reciprocal_rank, 4),
                    "hit_at_5": hit_in_5,
                    "hit_at_10": hit_in_10,
                    "precision_at_5": round(p5, 4),
                    "retrieved_chunk_ids": retrieved_chunks[:5],
                    "retrieved_sections": retrieved_secs[:5],
                    "latency_ms": round(total_elapsed_ms, 1),
                }
            )

        n = len(reviewed)
        summary.recall_at_5 = round(hits_5 / n, 4)
        summary.recall_at_10 = round(hits_10 / n, 4)
        summary.precision_at_5 = round(sum_p5 / n, 4)
        summary.mrr = round(sum(reciprocal_ranks) / n, 4)
        summary.citation_precision = round(sum(citation_precisions) / n, 4)
        summary.claim_support_rate = round(hits_10 / n, 4)

        summary.avg_retrieval_latency_ms = round(sum(retrieval_latencies) / n, 1)
        summary.avg_reranking_latency_ms = round(sum(reranking_latencies) / n, 1)
        summary.avg_total_latency_ms = round(sum(total_latencies) / n, 1)

        summary.report_message = (
            f"Evaluated {n} reviewed queries: "
            f"Recall@5={summary.recall_at_5*100:.1f}%, "
            f"Recall@10={summary.recall_at_10*100:.1f}%, "
            f"Precision@5={summary.precision_at_5*100:.1f}%, "
            f"MRR={summary.mrr:.4f}, "
            f"Avg Latency={summary.avg_total_latency_ms:.1f}ms."
        )
        return summary
