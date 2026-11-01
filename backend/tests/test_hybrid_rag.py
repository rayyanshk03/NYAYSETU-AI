"""Tests for Hybrid RAG: BM25, RRF Fusion, Cross-Encoder Reranker, Citation Validator, and Audit.

All tests operate strictly on synthetic or temporary mock fixtures.
No fake legal provisions or claims of legal currency are asserted.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion.chunk import LegalChunk
from app.rag.bm25_retriever import BM25LegalRetriever, _tokenize_legal
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import LegalReranker
from app.rag.citation_validator import CitationValidator, CitationValidationResult
from app.rag.audit import CorpusAuditor, CorpusAuditReport
from app.rag.evaluation import LegalRAGEvaluator, EvalQuery
from app.rag.vector_store import ChromaVectorStore
from app.schemas.retrieval import RetrievedPassage


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_records():
    """Create synthetic test chunk records for BM25 and hybrid tests."""
    return [
        {
            "chunk_id": "synth_hma_13b",
            "legal_text": "Section 13B. Divorce by mutual consent. Both parties may jointly petition for divorce after living separately for one year.",
            "metadata": {
                "document_id": "A1955-25",
                "document_title": "Hindu Marriage Act, 1955",
                "act_name": "Hindu Marriage Act, 1955",
                "section_number": "13B",
                "section_title": "Divorce by mutual consent",
                "source_url": "https://www.indiacode.nic.in/handle/123456789/1567",
                "page_start": 12,
                "page_end": 12,
            },
        },
        {
            "chunk_id": "synth_crpc_125",
            "legal_text": "Section 125. Order for maintenance of wives, children and parents by any person having sufficient means.",
            "metadata": {
                "document_id": "A1974-02",
                "document_title": "Code of Criminal Procedure, 1973",
                "act_name": "Code of Criminal Procedure, 1973",
                "section_number": "125",
                "section_title": "Order for maintenance",
                "source_url": "https://www.indiacode.nic.in/handle/123456789/1600",
                "page_start": 45,
                "page_end": 46,
            },
        },
        {
            "chunk_id": "synth_dv_12",
            "legal_text": "Section 12. Application to Magistrate for obtaining orders of relief including protection orders and residence orders.",
            "metadata": {
                "document_id": "A2005-43",
                "document_title": "Protection of Women from Domestic Violence Act, 2005",
                "act_name": "Protection of Women from Domestic Violence Act, 2005",
                "section_number": "12",
                "section_title": "Application to Magistrate",
                "source_url": "https://www.indiacode.nic.in/handle/123456789/2005",
                "page_start": 8,
                "page_end": 9,
            },
        },
    ]


@pytest.fixture
def temp_vector_store(synthetic_records):
    """Temporary ChromaVectorStore populated with synthetic records."""
    temp_dir = Path(tempfile.mkdtemp(prefix="nyayasetu_hybrid_test_"))
    store = ChromaVectorStore(
        persist_dir=temp_dir,
        collection_name="test_hybrid_corpus",
    )
    chunks = [
        LegalChunk(
            chunk_id=r["chunk_id"],
            document_id=r["metadata"]["document_id"],
            document_title=r["metadata"]["document_title"],
            act_name=r["metadata"]["act_name"],
            section_number=r["metadata"]["section_number"],
            section_title=r["metadata"]["section_title"],
            page_start=r["metadata"]["page_start"],
            page_end=r["metadata"]["page_end"],
            source_url=r["metadata"]["source_url"],
            legal_text=r["legal_text"],
            content_hash=f"hash_{r['chunk_id']}",
            char_count=len(r["legal_text"]),
            token_estimate=len(r["legal_text"]) // 4,
        )
        for r in synthetic_records
    ]
    store.index_chunks(chunks, show_progress=False)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def empty_vector_store():
    """Empty temporary ChromaVectorStore."""
    temp_dir = Path(tempfile.mkdtemp(prefix="nyayasetu_empty_test_"))
    store = ChromaVectorStore(
        persist_dir=temp_dir,
        collection_name="test_empty_corpus",
    )
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


# ── BM25 Tests ───────────────────────────────────────────────────────────────

def test_bm25_legal_tokenizer():
    """Verify BM25 tokenizer properly extracts legal section alphanumeric tokens."""
    tokens = _tokenize_legal("Section 13B of the Hindu Marriage Act, 1955: mutual consent divorce")
    assert "13b" in tokens
    assert "section" in tokens
    assert "divorce" in tokens
    assert "1955" in tokens
    assert "hindu" in tokens
    assert "marriage" in tokens


def test_bm25_index_and_search(synthetic_records):
    """Verify BM25 index creation and search precision."""
    retriever = BM25LegalRetriever()
    retriever.build_index(synthetic_records)
    assert retriever.is_built
    assert retriever.corpus_size == 3

    # Query for mutual consent divorce should rank HMA 13B first
    results = retriever.search("mutual consent divorce Section 13B", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "synth_hma_13b"
    assert results[0]["bm25_score"] > 0.0


def test_bm25_document_id_filter(synthetic_records):
    """Verify BM25 filtering by document_id."""
    retriever = BM25LegalRetriever()
    retriever.build_index(synthetic_records)

    # Search for maintenance with filter restricted to Domestic Violence Act
    results = retriever.search("maintenance", top_k=5, document_id="A2005-43")
    for r in results:
        assert r["metadata"]["document_id"] == "A2005-43"


# ── Hybrid Retriever Tests ───────────────────────────────────────────────────

def test_hybrid_retriever_empty_corpus(empty_vector_store):
    """Verify hybrid retriever handles empty corpus gracefully without throwing."""
    hybrid = HybridRetriever(vector_store=empty_vector_store, reranker=LegalReranker(enabled=False))
    resp = hybrid.retrieve("grounds for divorce", top_k=5)
    assert resp.insufficient_evidence is True
    assert resp.total_results == 0
    assert len(resp.passages) == 0


def test_hybrid_retriever_empty_query(temp_vector_store):
    """Verify hybrid retriever handles blank query gracefully."""
    hybrid = HybridRetriever(vector_store=temp_vector_store, reranker=LegalReranker(enabled=False))
    resp = hybrid.retrieve("   ", top_k=5)
    assert resp.insufficient_evidence is True
    assert resp.total_results == 0


def test_hybrid_retriever_rrf_and_deduplication(temp_vector_store, synthetic_records):
    """Verify RRF fusion combines dense and sparse results without duplicate chunk IDs."""
    bm25 = BM25LegalRetriever()
    bm25.build_index(synthetic_records)

    hybrid = HybridRetriever(
        vector_store=temp_vector_store,
        bm25_retriever=bm25,
        reranker=LegalReranker(enabled=False),
    )
    resp = hybrid.retrieve("Section 13B mutual consent divorce", top_k=3, min_score=0.0)
    assert resp.total_results > 0
    chunk_ids = [p.chunk_id for p in resp.passages]
    # No duplicate chunk IDs allowed
    assert len(chunk_ids) == len(set(chunk_ids))
    # Latency tracking dictionary present
    assert "retrieval_latency_ms" in resp.passages[0].metadata


def test_hybrid_retriever_document_id_filter(temp_vector_store, synthetic_records):
    """Verify hybrid retrieval filters by document_id."""
    bm25 = BM25LegalRetriever()
    bm25.build_index(synthetic_records)

    hybrid = HybridRetriever(
        vector_store=temp_vector_store,
        bm25_retriever=bm25,
        reranker=LegalReranker(enabled=False),
    )
    resp = hybrid.retrieve("maintenance", top_k=5, document_id="A1974-02", min_score=0.0)
    for p in resp.passages:
        assert p.document_id == "A1974-02"


# ── Reranker Tests ───────────────────────────────────────────────────────────

def test_reranker_fallback_on_error():
    """Verify reranker falls back to original order on model failure without raising."""
    reranker = LegalReranker(enabled=True)
    # Simulate failed model load
    reranker._load_failed = True

    passages = [
        RetrievedPassage(
            chunk_id="p1",
            document_id="d1",
            document_title="Title 1",
            legal_text="First passage text",
            page_start=1,
            page_end=1,
            page_number=1,
            source_url="https://www.indiacode.nic.in/1",
            score=0.85,
        ),
        RetrievedPassage(
            chunk_id="p2",
            document_id="d2",
            document_title="Title 2",
            legal_text="Second passage text",
            page_start=2,
            page_end=2,
            page_number=2,
            source_url="https://www.indiacode.nic.in/2",
            score=0.75,
        ),
    ]

    result = reranker.rerank(query="test query", passages=passages, top_k=2)
    assert len(result) == 2
    assert result[0].chunk_id == "p1"
    assert result[1].chunk_id == "p2"


def test_reranker_mock_scoring():
    """Verify reranker reorders passages based on cross-encoder predictions."""
    reranker = LegalReranker(enabled=True)
    mock_model = MagicMock()
    # Mock scores so p2 gets higher score (5.0) than p1 (-1.0)
    mock_model.predict.return_value = [-1.0, 5.0]
    reranker._model = mock_model

    passages = [
        RetrievedPassage(
            chunk_id="p1",
            document_id="d1",
            document_title="Title 1",
            legal_text="First passage text",
            page_start=1,
            page_end=1,
            page_number=1,
            source_url="https://www.indiacode.nic.in/1",
            score=0.85,
        ),
        RetrievedPassage(
            chunk_id="p2",
            document_id="d2",
            document_title="Title 2",
            legal_text="Second passage text",
            page_start=2,
            page_end=2,
            page_number=2,
            source_url="https://www.indiacode.nic.in/2",
            score=0.75,
        ),
    ]

    result = reranker.rerank(query="divorce query", passages=passages, top_k=2)
    assert len(result) == 2
    # p2 should now be ranked first due to higher score
    assert result[0].chunk_id == "p2"
    assert result[0].score == 5.0
    assert result[0].metadata["reranker_score"] == 5.0


# ── Citation Validator Tests ─────────────────────────────────────────────────

def test_citation_validator_valid():
    """Verify validator marks matching citations as verified."""
    validator = CitationValidator()
    passages = [
        RetrievedPassage(
            chunk_id="chk_hma_13",
            document_id="A1955-25",
            document_title="Hindu Marriage Act, 1955",
            act_name="Hindu Marriage Act, 1955",
            section_number="13",
            legal_text="Divorce grounds...",
            page_start=10,
            page_end=10,
            page_number=10,
            source_url="https://www.indiacode.nic.in/handle/123456789/1567",
            score=0.9,
        )
    ]
    citations = [
        {
            "chunk_id": "chk_hma_13",
            "act_name": "Hindu Marriage Act, 1955",
            "section_number": "13",
            "source_url": "https://www.indiacode.nic.in/handle/123456789/1567",
        }
    ]
    results = validator.validate_citations(citations, passages)
    assert len(results) == 1
    assert results[0].verified is True
    assert len(results[0].failure_reasons) == 0


def test_citation_validator_fabricated_chunk():
    """Verify validator fails a citation with a fabricated chunk_id."""
    validator = CitationValidator()
    passages = [
        RetrievedPassage(
            chunk_id="chk_hma_13",
            document_id="A1955-25",
            document_title="Hindu Marriage Act, 1955",
            act_name="Hindu Marriage Act, 1955",
            section_number="13",
            legal_text="Divorce grounds...",
            page_start=10,
            page_end=10,
            page_number=10,
            source_url="https://www.indiacode.nic.in/handle/123456789/1567",
            score=0.9,
        )
    ]
    citations = [
        {
            "chunk_id": "fabricated_chunk_999",
            "act_name": "Hindu Marriage Act, 1955",
            "section_number": "13",
        }
    ]
    results = validator.validate_citations(citations, passages)
    assert results[0].verified is False
    assert any("not found in retrieved passages" in r for r in results[0].failure_reasons)


def test_citation_validator_untrusted_url():
    """Verify validator flags citations pointing to untrusted domains."""
    validator = CitationValidator()
    passages = [
        RetrievedPassage(
            chunk_id="chk_hma_13",
            document_id="A1955-25",
            document_title="Hindu Marriage Act, 1955",
            act_name="Hindu Marriage Act, 1955",
            section_number="13",
            legal_text="Divorce grounds...",
            page_start=10,
            page_end=10,
            page_number=10,
            source_url="https://www.indiacode.nic.in/handle/123456789/1567",
            score=0.9,
        )
    ]
    citations = [
        {
            "chunk_id": "chk_hma_13",
            "act_name": "Hindu Marriage Act, 1955",
            "section_number": "13",
            "source_url": "https://random-unverified-blog.com/legal-advice",
        }
    ]
    results = validator.validate_citations(citations, passages)
    assert results[0].verified is False
    assert any("trusted" in r.lower() for r in results[0].failure_reasons)


# ── Corpus Auditor Tests ─────────────────────────────────────────────────────

def test_corpus_auditor_runs_on_workspace():
    """Verify CorpusAuditor runs on workspace directories without unhandled exceptions."""
    auditor = CorpusAuditor()
    report = auditor.run()
    assert isinstance(report, CorpusAuditReport)
    assert report.total_raw_files >= 0
    assert report.total_chunks_in_jsonl >= 0
    # Mandatory disclaimer must be present in notices
    assert any("amendment status notice" in n.lower() or "snapshot" in n.lower() for n in report.notices)


# ── Evaluation Framework Tests ───────────────────────────────────────────────

def test_evaluator_unreviewed_policy():
    """Verify evaluator reports insufficient ground truth when queries are unreviewed."""
    evaluator = LegalRAGEvaluator()
    unreviewed_queries = [
        EvalQuery(
            query_id="q1",
            query="divorce mutual consent",
            expected_document_ids=["A1955-25"],
            expected_section_ids=["13B"],
            reviewer_status="unreviewed",
        )
    ]
    metrics = evaluator.evaluate(unreviewed_queries)
    assert metrics.has_sufficient_ground_truth is False
    assert metrics.reviewed_queries == 0
    assert "No expert-reviewed ground-truth labels" in metrics.report_message


def test_evaluator_reviewed_metrics():
    """Verify evaluator computes recall and MRR correctly on reviewed synthetic queries."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = MagicMock(
        passages=[
            RetrievedPassage(
                chunk_id="c1",
                document_id="A1955-25",
                document_title="Hindu Marriage Act, 1955",
                section_number="13B",
                legal_text="Section 13B mutual consent divorce",
                page_start=1,
                page_end=1,
                page_number=1,
                source_url="https://www.indiacode.nic.in/1",
                score=0.9,
                metadata={"retrieval_latency_ms": {"dense_ms": 10.0, "sparse_ms": 5.0, "rerank_ms": 8.0}},
            )
        ]
    )

    evaluator = LegalRAGEvaluator(retriever=mock_retriever)
    reviewed_queries = [
        EvalQuery(
            query_id="q1",
            query="divorce mutual consent",
            expected_document_ids=["A1955-25"],
            expected_section_ids=["13B"],
            reviewer_status="reviewed",
        )
    ]
    metrics = evaluator.evaluate(reviewed_queries, top_k=5)
    assert metrics.has_sufficient_ground_truth is True
    assert metrics.reviewed_queries == 1
    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.precision_at_5 == 0.2  # 1 hit out of 5


# ── API Endpoint Tests ───────────────────────────────────────────────────────

def test_api_retrieve_hybrid_endpoint():
    """Verify /api/retrieve returns structured hybrid retrieval schema."""
    client = TestClient(app)
    response = client.post(
        "/api/retrieve",
        json={"query": "grounds for divorce", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "passages" in data
    assert "insufficient_evidence" in data
    assert "total_results" in data


def test_api_corpus_status_and_audit():
    """Verify /api/corpus/status and /api/corpus/audit endpoints."""
    client = TestClient(app)

    # Status
    status_resp = client.get("/api/corpus/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert "indexed_provisions" in status_data
    assert "hybrid_retrieval_enabled" in status_data
    assert status_data["hybrid_retrieval_enabled"] is True

    # Audit
    audit_resp = client.post("/api/corpus/audit")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert "total_raw_files" in audit_data
    assert "notices" in audit_data
