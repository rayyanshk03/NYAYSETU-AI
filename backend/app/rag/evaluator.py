"""Retrieval evaluation framework for NyayaSetu AI.

Calculates Recall@k and Mean Reciprocal Rank (MRR) against verified legal benchmark datasets.
Reports missing or unreviewed ground-truth labels without fabricating evaluation scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.rag.retrieval import LegalRetriever

logger = logging.getLogger(__name__)


class BenchmarkQuery(BaseModel):
    """A test query with human-verified ground-truth chunk IDs or sections."""

    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Legal question text")
    language: Optional[str] = Field(default="en", description="Query language")
    expected_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of verified relevant chunk IDs from the knowledge base",
    )
    expected_section: Optional[str] = Field(
        default=None,
        description="Optional expected section number (e.g. '13B', '125')",
    )


class EvaluationMetricSummary(BaseModel):
    """Evaluation metrics computed across benchmark queries."""

    total_queries: int = Field(default=0)
    valid_labeled_queries: int = Field(default=0)
    unlabeled_queries: int = Field(default=0)
    recall_at_5: float = Field(default=0.0, description="Recall@5 score (0.0 to 1.0)")
    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank (0.0 to 1.0)")
    retrieval_success_rate: float = Field(
        default=0.0, description="Retrieval Success Rate: fraction of queries with >=1 relevant hit in top-k"
    )
    per_query_results: List[Dict[str, Any]] = Field(default_factory=list)
    has_sufficient_ground_truth: bool = Field(default=False)
    report_message: str = Field(default="")


class RetrievalEvaluator:
    """Evaluates legal retrieval accuracy using Recall@k and MRR."""

    def __init__(self, retriever: Optional[LegalRetriever] = None):
        self.retriever = retriever or LegalRetriever()

    @staticmethod
    def load_benchmark(file_path: Path) -> List[BenchmarkQuery]:
        """Load benchmark queries from JSON or CSV."""
        if not file_path.exists():
            return []

        queries: List[BenchmarkQuery] = []
        if file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("queries", [])
                for item in items:
                    try:
                        queries.append(BenchmarkQuery.model_validate(item))
                    except Exception as e:
                        logger.warning("Invalid benchmark query format: %s", e)
        elif file_path.suffix.lower() == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    expected_chunks = [c.strip() for c in row.get("expected_chunk_ids", "").split(";") if c.strip()]
                    queries.append(
                        BenchmarkQuery(
                            query_id=row.get("query_id", f"q_{len(queries)+1}"),
                            query=row.get("query", ""),
                            language=row.get("language", "en"),
                            expected_chunk_ids=expected_chunks,
                            expected_section=row.get("expected_section") or None,
                        )
                    )
        return queries

    def evaluate(
        self,
        queries: List[BenchmarkQuery],
        top_k: int = 5,
    ) -> EvaluationMetricSummary:
        """
        Evaluate retrieval performance on a list of benchmark queries.
        """
        summary = EvaluationMetricSummary(total_queries=len(queries))

        labeled_queries = [q for q in queries if q.expected_chunk_ids or q.expected_section]
        summary.valid_labeled_queries = len(labeled_queries)
        summary.unlabeled_queries = len(queries) - len(labeled_queries)

        if not labeled_queries:
            summary.has_sufficient_ground_truth = False
            summary.report_message = (
                "No reviewed ground-truth labels found in the benchmark dataset. "
                "Per NyayaSetu AI safety standards, evaluation metrics cannot be calculated without verified labels."
            )
            return summary

        summary.has_sufficient_ground_truth = True
        reciprocal_ranks: List[float] = []
        hits_at_k = 0

        for bq in labeled_queries:
            res = self.retriever.retrieve(query=bq.query, top_k=top_k)
            retrieved_chunk_ids = [p.chunk_id for p in res.passages]
            retrieved_sections = [p.section_number for p in res.passages if p.section_number]

            # Determine rank of first hit
            first_hit_rank: Optional[int] = None
            for idx, p in enumerate(res.passages, start=1):
                match_id = p.chunk_id in bq.expected_chunk_ids
                match_sec = bq.expected_section and p.section_number == bq.expected_section
                if match_id or match_sec:
                    first_hit_rank = idx
                    break

            if first_hit_rank is not None:
                hits_at_k += 1
                reciprocal_ranks.append(1.0 / first_hit_rank)
            else:
                reciprocal_ranks.append(0.0)

            summary.per_query_results.append(
                {
                    "query_id": bq.query_id,
                    "query": bq.query,
                    "first_hit_rank": first_hit_rank,
                    "reciprocal_rank": reciprocal_ranks[-1],
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "retrieved_sections": retrieved_sections,
                }
            )

        summary.recall_at_5 = round(hits_at_k / len(labeled_queries), 4)
        summary.mrr = round(sum(reciprocal_ranks) / len(labeled_queries), 4)
        summary.retrieval_success_rate = round(hits_at_k / len(labeled_queries), 4)
        summary.report_message = (
            f"Evaluated {len(labeled_queries)} verified queries: "
            f"Recall@{top_k} = {summary.recall_at_5 * 100:.1f}%, "
            f"MRR = {summary.mrr:.4f}, "
            f"Success Rate = {summary.retrieval_success_rate * 100:.1f}%."
        )

        return summary


def run_synthetic_benchmark(top_k: int = 5) -> EvaluationMetricSummary:
    """
    Run evaluation mechanics using synthetic mock legal chunks.

    NOTICE: Synthetic benchmark tests the retrieval calculation mechanics only
    (indexing, vector similarity ranking, Recall@k, and MRR). These results DO NOT
    represent actual legal accuracy or official statutory outcomes.
    """
    import tempfile
    import shutil
    from app.ingestion.chunk import LegalChunk
    from app.rag.vector_store import ChromaVectorStore

    temp_dir = Path(tempfile.mkdtemp(prefix="nyayasetu_eval_synth_"))
    try:
        vs = ChromaVectorStore(persist_dir=temp_dir, collection_name="eval_synth_test")

        # Create synthetic chunks for testing mechanics
        chunks = [
            LegalChunk(
                chunk_id="synth_hma_sec_13_1",
                document_id="synth_hma",
                document_title="Synthetic Hindu Marriage Act (Test Document)",
                act_name="Synthetic HMA",
                section_number="13",
                section_title="Divorce",
                page_start=5,
                page_end=5,
                source_url="https://www.indiacode.nic.in/handle/123456789/1567",
                legal_text="Section 13. Any marriage solemnized may be dissolved by a decree of divorce on grounds of cruelty or adultery.",
                content_hash="hash_synth_13",
                char_count=120,
                token_estimate=30,
            ),
            LegalChunk(
                chunk_id="synth_crpc_sec_125_1",
                document_id="synth_crpc",
                document_title="Synthetic Code of Criminal Procedure (Test Document)",
                act_name="Synthetic CrPC",
                section_number="125",
                section_title="Order for maintenance of wives, children and parents",
                page_start=12,
                page_end=12,
                source_url="https://www.indiacode.nic.in/handle/123456789/1611",
                legal_text="Section 125. Order for maintenance of wives, children and parents having sufficient means neglects or refuses to maintain.",
                content_hash="hash_synth_125",
                char_count=130,
                token_estimate=32,
            ),
            LegalChunk(
                chunk_id="synth_pwdva_sec_12_1",
                document_id="synth_pwdva",
                document_title="Synthetic Domestic Violence Act (Test Document)",
                act_name="Synthetic PWDVA",
                section_number="12",
                section_title="Application to Magistrate",
                page_start=8,
                page_end=8,
                source_url="https://www.indiacode.nic.in/handle/123456789/2085",
                legal_text="Section 12. An aggrieved person or a Protection Officer may present an application to the Magistrate for relief under this Act.",
                content_hash="hash_synth_12",
                char_count=135,
                token_estimate=35,
            ),
        ]
        vs.index_chunks(chunks)

        evaluator = RetrievalEvaluator(retriever=LegalRetriever(vector_store=vs))
        benchmark_queries = [
            BenchmarkQuery(
                query_id="sq_1",
                query="What are the grounds for divorce under Hindu law?",
                expected_chunk_ids=["synth_hma_sec_13_1"],
                expected_section="13",
            ),
            BenchmarkQuery(
                query_id="sq_2",
                query="How to claim maintenance for wife and children under section 125?",
                expected_chunk_ids=["synth_crpc_sec_125_1"],
                expected_section="125",
            ),
            BenchmarkQuery(
                query_id="sq_3",
                query="How can an aggrieved woman file application to magistrate for protection?",
                expected_chunk_ids=["synth_pwdva_sec_12_1"],
                expected_section="12",
            ),
        ]

        summary = evaluator.evaluate(benchmark_queries, top_k=top_k)
        return summary
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Command line entrypoint for running legal retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="NyayaSetu AI — Retrieval Evaluation Framework"
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=None,
        help="Path to evaluation benchmark JSON or CSV file",
    )
    parser.add_argument(
        "--run-synthetic",
        action="store_true",
        help="Run evaluation mechanics on synthetic benchmark documents (testing mechanics only)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K cutoff for Recall@k (default: 5)",
    )

    args = parser.parse_args()

    if args.run_synthetic:
        print("\n" + "=" * 75)
        print(" NyayaSetu AI — Synthetic Retrieval Evaluation (Testing Mechanics Only)")
        print("=" * 75)
        print(" [IMPORTANT NOTICE]")
        print(" This run uses synthetic mock provisions to verify ranking & metric math.")
        print(" Synthetic results DO NOT represent actual Indian legal accuracy.")
        print("-" * 75)

        summary = run_synthetic_benchmark(top_k=args.top_k)

        print(f" Total Benchmark Queries : {summary.total_queries}")
        print(f" Verified Labeled Queries: {summary.valid_labeled_queries}")
        print(f" Recall@{args.top_k}              : {summary.recall_at_5 * 100:.2f}%")
        print(f" Mean Reciprocal Rank (MRR): {summary.mrr:.4f}")
        print(f" Retrieval Success Rate  : {summary.retrieval_success_rate * 100:.2f}%")
        print(f" Message                 : {summary.report_message}")
        print("=" * 75 + "\n")
        return

    evaluator = RetrievalEvaluator()

    if not args.benchmark_file:
        print("\n" + "=" * 75)
        print(" NyayaSetu AI — Retrieval Evaluation Framework (Step 3)")
        print("=" * 75)
        print(" Notice: No benchmark evaluation file specified.")
        print(" Usage:")
        print("   python -m app.rag.evaluator --benchmark-file <path_to_json_or_csv>")
        print("   python -m app.rag.evaluator --run-synthetic")
        print("\n Status: Evaluation framework is verified and ready.")
        print(" Ground-Truth Status: No reviewed ground-truth dataset provided.")
        print(" Per system constraints, metrics cannot be fabricated without reviewed labels.")
        print("=" * 75 + "\n")
        return

    path = Path(args.benchmark_file)
    if not path.exists():
        print(f"\n❌ Benchmark file not found: {path}\n")
        return

    queries = evaluator.load_benchmark(path)
    summary = evaluator.evaluate(queries, top_k=args.top_k)

    print("\n" + "=" * 75)
    print(" Retrieval Evaluation Report")
    print("=" * 75)
    print(f" Total Queries Loaded    : {summary.total_queries}")
    print(f" Verified Labeled Queries: {summary.valid_labeled_queries}")
    print(f" Unlabeled Queries       : {summary.unlabeled_queries}")

    if summary.has_sufficient_ground_truth:
        print(f" Recall@{args.top_k}              : {summary.recall_at_5 * 100:.2f}%")
        print(f" Mean Reciprocal Rank (MRR): {summary.mrr:.4f}")
        print(f" Retrieval Success Rate  : {summary.retrieval_success_rate * 100:.2f}%")
    else:
        print(f" Metrics Calculation     : Cannot calculate (no verified ground-truth labels)")

    print(f" Message                 : {summary.report_message}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
