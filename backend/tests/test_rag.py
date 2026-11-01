"""Unit and integration tests for Step 3: RAG Retrieval Engine.

Covers:
1. Embedding generation (dimensions, normalization, multilingual encoding, mock fallback)
2. Vector indexing with ChromaDB
3. Repeated indexing (idempotency, upsert, update behavior)
4. Metadata preservation across all statutory fields
5. Top-k retrieval with ranking and score thresholding
6. Empty queries validation
7. Empty collections graceful handling
8. API request validation (Pydantic constraints on /api/retrieve)
9. Retrieval response structure and disclaimer verification

Uses synthetic test documents only. Does not fabricate legal text or claims of accuracy.
"""

import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ingestion.chunk import LegalChunk
from app.rag.embeddings import MultilingualEmbedder
from app.rag.vector_store import ChromaVectorStore
from app.rag.retrieval import LegalRetriever, retrieve_legal_passages
from app.rag.indexer import LegalCorpusIndexer
from app.schemas.chat import SupportedLanguage
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievedPassage


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_vector_store():
    """Create an isolated, temporary ChromaVectorStore for testing mechanics."""
    temp_dir = Path(tempfile.mkdtemp(prefix="nyayasetu_test_chroma_"))
    store = ChromaVectorStore(
        persist_dir=temp_dir,
        collection_name="test_legal_corpus",
    )
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def synthetic_chunks():
    """Create synthetic test chunks covering diverse family law topics."""
    return [
        LegalChunk(
            chunk_id="synth_hma_13_1",
            document_id="hma_1955",
            document_title="Hindu Marriage Act, 1955",
            act_name="Hindu Marriage Act, 1955",
            section_number="13",
            section_title="Divorce",
            page_start=6,
            page_end=7,
            source_url="https://www.indiacode.nic.in/handle/123456789/1567",
            publication_date="1955-05-18",
            effective_date="1955-05-18",
            document_version="As modified up to date",
            legal_text=(
                "Section 13. Divorce.—(1) Any marriage solemnized may, on a petition "
                "presented by either the husband or the wife, be dissolved by a decree "
                "of divorce on the ground that the other party has treated the petitioner with cruelty."
            ),
            content_hash="hash_synth_hma_13",
            char_count=220,
            token_estimate=50,
        ),
        LegalChunk(
            chunk_id="synth_crpc_125_1",
            document_id="crpc_sec_125",
            document_title="Code of Criminal Procedure, 1973",
            act_name="Code of Criminal Procedure, 1973",
            section_number="125",
            section_title="Order for maintenance of wives, children and parents",
            page_start=15,
            page_end=16,
            source_url="https://www.indiacode.nic.in/handle/123456789/1611",
            publication_date="1974-01-25",
            effective_date="1974-04-01",
            document_version="As modified up to date",
            legal_text=(
                "Section 125. Order for maintenance of wives, children and parents.—"
                "(1) If any person having sufficient means neglects or refuses to maintain "
                "his wife, unable to maintain herself, a Magistrate may order monthly maintenance allowance."
            ),
            content_hash="hash_synth_crpc_125",
            char_count=235,
            token_estimate=55,
        ),
        LegalChunk(
            chunk_id="synth_pwdva_12_1",
            document_id="pwdva_2005",
            document_title="Protection of Women from Domestic Violence Act, 2005",
            act_name="Protection of Women from Domestic Violence Act, 2005",
            section_number="12",
            section_title="Application to Magistrate",
            page_start=9,
            page_end=10,
            source_url="https://www.indiacode.nic.in/handle/123456789/2085",
            publication_date="2005-09-13",
            effective_date="2006-10-26",
            document_version="As modified up to date",
            legal_text=(
                "Section 12. Application to Magistrate.—(1) An aggrieved person or a Protection "
                "Officer or any other person on behalf of the aggrieved person may present an "
                "application to the Magistrate seeking one or more reliefs under this Act."
            ),
            content_hash="hash_synth_pwdva_12",
            char_count=240,
            token_estimate=58,
        ),
        LegalChunk(
            chunk_id="synth_gwa_17_1",
            document_id="gwa_1890",
            document_title="Guardians and Wards Act, 1890",
            act_name="Guardians and Wards Act, 1890",
            section_number="17",
            section_title="Matters to be considered by the Court in appointing guardian",
            page_start=11,
            page_end=11,
            source_url="https://www.indiacode.nic.in/handle/123456789/2296",
            publication_date="1890-03-21",
            effective_date="1890-07-01",
            document_version="As modified up to date",
            legal_text=(
                "Section 17. Matters to be considered by the Court in appointing guardian.—"
                "(1) In appointing or declaring the guardian of a minor, the Court shall, "
                "subject to the provisions of this section, be guided by what, consistently "
                "with the law to which the minor is subject, appears to be for the welfare of the minor."
            ),
            content_hash="hash_synth_gwa_17",
            char_count=270,
            token_estimate=65,
        ),
    ]


# ── 1. Embedding Generation Tests ────────────────────────────────────────────

class TestEmbeddingGeneration:
    """Test embedding model loading, dimensionality, normalization, and multilinguality."""

    def test_embedding_dimensions(self):
        embedder = MultilingualEmbedder.get_instance()
        vec = embedder.encode_text("What are the grounds for divorce?")
        assert len(vec) == 384
        assert isinstance(vec, list)
        assert all(isinstance(x, float) for x in vec)

    def test_empty_string_embedding(self):
        embedder = MultilingualEmbedder.get_instance()
        vec = embedder.encode_text("")
        assert len(vec) == 384
        assert all(x == 0.0 for x in vec)

    def test_embedding_normalization(self):
        import numpy as np

        embedder = MultilingualEmbedder.get_instance()
        vec = embedder.encode_text("Divorce grounds under Hindu Marriage Act Section 13")
        norm = np.linalg.norm(vec)
        # Normalized vectors must have L2 norm approximately 1.0 (unit sphere)
        assert abs(norm - 1.0) < 1e-4

    def test_multilingual_encoding_without_errors(self):
        embedder = MultilingualEmbedder.get_instance()
        # English
        vec_en = embedder.encode_text("Maintenance rights of a wife")
        # Hindi
        vec_hi = embedder.encode_text("पत्नी के भरण-पोषण के कानूनी अधिकार क्या हैं?")
        # Kannada
        vec_kn = embedder.encode_text("ಪತ್ನಿಯ ಜೀವನಾಂಶ ಹಕ್ಕುಗಳು ಯಾವುವು?")

        assert len(vec_en) == 384
        assert len(vec_hi) == 384
        assert len(vec_kn) == 384

    def test_batch_encoding(self):
        embedder = MultilingualEmbedder.get_instance()
        passages = [
            "Section 13. Grounds for divorce.",
            "Section 125. Order for maintenance of wives.",
            "Section 12. Protection orders for domestic violence.",
        ]
        batch_vecs = embedder.encode_passages(passages, batch_size=2)
        assert len(batch_vecs) == 3
        for vec in batch_vecs:
            assert len(vec) == 384

    def test_mock_fallback_on_demand(self):
        mock_embedder = MultilingualEmbedder(use_mock=True)
        vec = mock_embedder.encode_text("Test query with mock embedder")
        assert len(vec) == 384


# ── 2. Vector Indexing Tests ─────────────────────────────────────────────────

class TestVectorIndexing:
    """Test indexing synthetic chunks into ChromaDB."""

    def test_index_synthetic_chunks(self, temp_vector_store, synthetic_chunks):
        indexed = temp_vector_store.index_chunks(synthetic_chunks)
        assert indexed == len(synthetic_chunks)
        assert temp_vector_store.count() == len(synthetic_chunks)

    def test_index_empty_list(self, temp_vector_store):
        indexed = temp_vector_store.index_chunks([])
        assert indexed == 0
        assert temp_vector_store.count() == 0


# ── 3. Repeated Indexing Tests ───────────────────────────────────────────────

class TestRepeatedIndexing:
    """Test idempotency, upsert semantics, and update capabilities."""

    def test_repeated_indexing_does_not_duplicate(self, temp_vector_store, synthetic_chunks):
        # First index run
        temp_vector_store.index_chunks(synthetic_chunks)
        assert temp_vector_store.count() == len(synthetic_chunks)

        # Second index run with identical chunk IDs
        temp_vector_store.index_chunks(synthetic_chunks)
        assert temp_vector_store.count() == len(synthetic_chunks)

    def test_update_existing_chunk(self, temp_vector_store, synthetic_chunks):
        temp_vector_store.index_chunks(synthetic_chunks)

        # Update the legal text for the divorce chunk
        updated_chunk = LegalChunk(
            chunk_id=synthetic_chunks[0].chunk_id,
            document_id=synthetic_chunks[0].document_id,
            document_title=synthetic_chunks[0].document_title,
            act_name=synthetic_chunks[0].act_name,
            section_number="13B",
            section_title="Divorce by mutual consent",
            page_start=8,
            page_end=8,
            source_url=synthetic_chunks[0].source_url,
            legal_text="Section 13B. Divorce by mutual consent.—Both parties may jointly petition for divorce.",
            content_hash="updated_hash_13b",
            char_count=100,
            token_estimate=25,
        )

        temp_vector_store.index_chunks([updated_chunk])
        assert temp_vector_store.count() == len(synthetic_chunks)

        # Query and ensure updated content is returned
        results = temp_vector_store.query("mutual consent divorce", top_k=1)
        assert len(results) >= 1
        assert "Section 13B" in results[0]["legal_text"]

    def test_explicit_reset_collection(self, temp_vector_store, synthetic_chunks):
        temp_vector_store.index_chunks(synthetic_chunks)
        assert temp_vector_store.count() > 0

        temp_vector_store.reset_collection()
        assert temp_vector_store.count() == 0


# ── 4. Metadata Preservation Tests ───────────────────────────────────────────

class TestMetadataPreservation:
    """Verify that all statutory metadata fields are preserved upon retrieval."""

    def test_all_metadata_fields_stored(self, temp_vector_store, synthetic_chunks):
        temp_vector_store.index_chunks(synthetic_chunks)

        results = temp_vector_store.query("cruelty and divorce", top_k=1)
        assert len(results) == 1
        top_meta = results[0]["metadata"]

        assert top_meta["document_id"] == "hma_1955"
        assert top_meta["document_title"] == "Hindu Marriage Act, 1955"
        assert top_meta["section_number"] == "13"
        assert top_meta["section_title"] == "Divorce"
        assert int(top_meta["page_start"]) == 6
        assert "indiacode.nic.in" in top_meta["source_url"]
        assert top_meta["publication_date"] == "1955-05-18"


# ── 5. Top-K Retrieval Tests ─────────────────────────────────────────────────

class TestTopKRetrieval:
    """Test top-k cutoff, similarity ranking, and reusable retrieval function."""

    def test_top_k_limits_results(self, temp_vector_store, synthetic_chunks):
        temp_vector_store.index_chunks(synthetic_chunks)
        retriever = LegalRetriever(vector_store=temp_vector_store)

        res = retriever.retrieve("divorce and maintenance rights", top_k=2)
        assert len(res.passages) <= 2
        assert res.total_results <= 2

    def test_reusable_function_returns_expected_fields(self, temp_vector_store, synthetic_chunks):
        temp_vector_store.index_chunks(synthetic_chunks)

        # Call retrieve_legal_passages directly
        res = retrieve_legal_passages(
            query="grounds for divorce",
            top_k=3,
            vector_store=temp_vector_store,
        )

        assert isinstance(res, RetrievalResponse)
        assert res.status == "success"
        assert len(res.passages) > 0

        first = res.passages[0]
        assert first.chunk_id is not None
        assert first.legal_text is not None
        assert first.document_title is not None
        assert first.act_name is not None
        assert first.section_number is not None
        assert first.page_number is not None
        assert first.source_url is not None
        assert isinstance(first.score, float)
        assert 0.0 <= first.score <= 1.0
        assert isinstance(first.metadata, dict)

    def test_metadata_filter_by_document(self, temp_vector_store, synthetic_chunks):
        temp_vector_store.index_chunks(synthetic_chunks)
        retriever = LegalRetriever(vector_store=temp_vector_store)

        # Restrict specifically to Domestic Violence Act
        res = retriever.retrieve(
            query="application to magistrate",
            top_k=5,
            document_id="pwdva_2005",
        )
        assert len(res.passages) > 0
        for p in res.passages:
            assert p.document_id == "pwdva_2005"


# ── 6. Empty Queries Tests ───────────────────────────────────────────────────

class TestEmptyQueries:
    """Verify validation and graceful handling for empty or whitespace-only queries."""

    def test_empty_string_retrieval(self, temp_vector_store):
        retriever = LegalRetriever(vector_store=temp_vector_store)
        res = retriever.retrieve("")
        assert res.insufficient_evidence is True
        assert res.total_results == 0
        assert res.passages == []
        assert "cannot be empty" in (res.evidence_note or "").lower()

    def test_whitespace_query_retrieval(self, temp_vector_store):
        retriever = LegalRetriever(vector_store=temp_vector_store)
        res = retriever.retrieve("     \n\t  ")
        assert res.insufficient_evidence is True
        assert res.total_results == 0


# ── 7. Empty Collections Tests ───────────────────────────────────────────────

class TestEmptyCollections:
    """Verify that querying an empty collection returns clear status without crashing."""

    def test_empty_collection_returns_clear_message(self, temp_vector_store):
        assert temp_vector_store.count() == 0
        retriever = LegalRetriever(vector_store=temp_vector_store)

        res = retriever.retrieve("What are the provisions for child custody?")
        assert res.insufficient_evidence is True
        assert res.total_results == 0
        assert res.passages == []
        assert "0 indexed provisions" in (res.evidence_note or "")


# ── 8. API Request Validation Tests ──────────────────────────────────────────

class TestApiRequestValidation:
    """Test Pydantic request validation on POST /api/retrieve."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_empty_query_rejected_422(self, client):
        response = client.post("/api/retrieve", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_field_rejected_422(self, client):
        response = client.post("/api/retrieve", json={"top_k": 5})
        assert response.status_code == 422

    def test_top_k_exceeding_max_rejected_422(self, client):
        response = client.post(
            "/api/retrieve",
            json={"query": "test query", "top_k": 50},  # Max allowed is 25
        )
        assert response.status_code == 422

    def test_top_k_below_min_rejected_422(self, client):
        response = client.post(
            "/api/retrieve",
            json={"query": "test query", "top_k": 0},
        )
        assert response.status_code == 422

    def test_invalid_language_code_rejected_422(self, client):
        response = client.post(
            "/api/retrieve",
            json={"query": "test query", "language": "fr"},  # Only en, hi, kn supported
        )
        assert response.status_code == 422


# ── 9. Retrieval Response Structure Tests ────────────────────────────────────

class TestRetrievalResponseStructure:
    """Test that the API endpoint returns the complete, typed response schema."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_valid_post_retrieve_response_structure(self, client):
        response = client.post(
            "/api/retrieve",
            json={
                "query": "What are the legal provisions related to maintenance?",
                "top_k": 5,
                "language": "en",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Check required fields of RetrievalResponse
        assert data["status"] == "success"
        assert data["query"] == "What are the legal provisions related to maintenance?"
        assert data["language"] == "en"
        assert "total_results" in data
        assert "insufficient_evidence" in data
        assert "passages" in data
        assert "timestamp" in data
        assert "disclaimer" in data
        # Mandatory disclaimer check
        assert "statutory provisions" in data["disclaimer"].lower()
        assert "similarity scores" in data["disclaimer"].lower()
