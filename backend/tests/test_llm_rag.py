"""Tests for Step 4: LLM-based legal answer generation using Retrieval-Augmented Generation.

Covers:
1. Successful grounded answer generation (mocked LLM & deterministic template fallback)
2. Missing evidence and insufficient evidence handling
3. Invalid citation detection and warning generation
4. Empty retrieval results and unindexed corpus handling
5. LLM provider failures (timeout, HTTP 500, network error) with graceful fallback
6. Emergency safety detection and official helpline routing (112, 181, 1091, 15100)
7. Full API response validation on POST /api/chat
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.agent.planner import QueryIntent, SupportedLanguage
from app.agent.synthesizer import LegalSynthesizer
from app.agent.safety import SafetyAuditor, safety_auditor, OFFICIAL_HELPLINES
from app.agent.orchestrator import LegalAgent
from app.services.llm_service import LLMService
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievedPassage


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_passages():
    """Create verified statutory passages for testing evidence grounding."""
    return [
        RetrievedPassage(
            chunk_id="hma_sec_13_1",
            document_id="hma_1955",
            document_title="Hindu Marriage Act, 1955",
            act_name="Hindu Marriage Act, 1955",
            section_number="13",
            section_title="Divorce",
            page_start=6,
            page_end=7,
            source_url="https://www.indiacode.nic.in/handle/123456789/1567",
            publication_date="1955-05-18",
            legal_text="Section 13. Any marriage solemnized may be dissolved by a decree of divorce on grounds of cruelty or desertion.",
            score=0.82,
        ),
        RetrievedPassage(
            chunk_id="crpc_sec_125_1",
            document_id="crpc_sec_125",
            document_title="Code of Criminal Procedure, 1973",
            act_name="Code of Criminal Procedure, 1973",
            section_number="125",
            section_title="Order for maintenance of wives, children and parents",
            page_start=15,
            page_end=16,
            source_url="https://www.indiacode.nic.in/handle/123456789/1611",
            publication_date="1974-01-25",
            legal_text="Section 125. Order for maintenance of wives, children and parents having sufficient means neglects or refuses to maintain.",
            score=0.74,
        ),
    ]


# ── 1. Grounded Answer Generation Tests ──────────────────────────────────────

class TestGroundedAnswerGeneration:
    """Test generating grounded answers via mocked LLM and deterministic template fallback."""

    def test_successful_grounded_answer_with_mocked_llm(self, sample_passages):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate.return_value = (
            "### Direct Explanation:\nUnder Section 13 of the Hindu Marriage Act, 1955, "
            "a spouse may petition for divorce on statutory grounds including cruelty [1].\n\n"
            "### Relevant Legal Provisions:\nHindu Marriage Act, 1955, Section 13.\n\n"
            "### Plain-Language Interpretation:\nCruelty constitutes conduct causing danger to life, limb, or mental health.\n\n"
            "### Important Limitations:\nPetition must be filed before the competent Family Court."
        )

        synthesizer = LegalSynthesizer(llm=mock_llm)
        result = synthesizer.synthesize_with_metadata(
            query="What are the grounds for divorce under Hindu law?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=sample_passages,
        )

        assert result.evidence_status == "sufficient"
        assert len(result.citations) == 2
        assert result.citations[0]["section_number"] == "13"
        assert "Hindu Marriage Act" in result.reply
        assert "Section 13" in result.reply
        # Disclaimer must always be ensured
        assert "Disclaimer" in result.reply or "legal advice" in result.reply.lower()

    def test_successful_grounded_answer_template_fallback(self, sample_passages):
        """When LLM provider returns None (e.g. template mode or offline), use structured fallback."""
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate.return_value = None  # Signals template fallback

        synthesizer = LegalSynthesizer(llm=mock_llm)
        result = synthesizer.synthesize_with_metadata(
            query="What are the grounds for divorce?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=sample_passages,
        )

        assert result.evidence_status == "sufficient"
        assert "Direct Explanation" in result.reply
        assert "Relevant Statutory Provisions" in result.reply
        assert "Plain-Language Interpretation" in result.reply
        assert "Important Limitations" in result.reply
        assert "Section 13" in result.reply


# ── 2. Missing Evidence & Sufficiency Tests ──────────────────────────────────

class TestMissingAndInsufficientEvidence:
    """Ensure system refuses to answer with general knowledge when evidence is absent."""

    def test_empty_passages_returns_insufficient_evidence(self):
        synthesizer = LegalSynthesizer()
        result = synthesizer.synthesize_with_metadata(
            query="What is the legal process for adoption?",
            intent=QueryIntent.ADOPTION,
            language=SupportedLanguage.ENGLISH,
            passages=[],
        )

        assert result.evidence_status == "insufficient"
        assert len(result.citations) == 0
        assert "Insufficient Evidence" in result.reply
        assert "No indexed statutory provisions met the minimum relevance threshold" in result.warnings[0]

    def test_low_similarity_passages_treated_as_insufficient(self, sample_passages):
        # Set scores below _MIN_CITE_SCORE (0.25)
        for p in sample_passages:
            p.score = 0.12

        synthesizer = LegalSynthesizer()
        result = synthesizer.synthesize_with_metadata(
            query="Irrelevant question",
            intent=QueryIntent.GENERAL,
            language=SupportedLanguage.ENGLISH,
            passages=sample_passages,
        )

        assert result.evidence_status == "insufficient"
        assert len(result.citations) == 0
        assert "Insufficient Evidence" in result.reply


# ── 3. Citation Validation Tests ─────────────────────────────────────────────

class TestCitationValidation:
    """Ensure the system detects and warns about hallucinated sections or citations."""

    def test_invalid_citation_detected_and_warned(self, sample_passages):
        mock_llm = MagicMock(spec=LLMService)
        # LLM references Section 498A and out-of-bounds citation [99], which do not exist in sample_passages
        mock_llm.generate.return_value = (
            "Under Section 498A [99], cruelty is punishable. "
            "Also Section 13 applies [1]."
        )

        synthesizer = LegalSynthesizer(llm=mock_llm)
        result = synthesizer.synthesize_with_metadata(
            query="Cruelty and divorce",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=sample_passages,
        )

        # Warnings should record ungrounded section 498A and out-of-bounds [99]
        warning_str = " ".join(result.warnings)
        assert "498A" in warning_str or "[99]" in warning_str

    def test_verified_citation_metadata(self, sample_passages):
        synthesizer = LegalSynthesizer()
        result = synthesizer.synthesize_with_metadata(
            query="divorce",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=sample_passages,
        )

        first_cite = result.citations[0]
        assert first_cite["verified"] is True
        assert first_cite["section_number"] == "13"
        assert "indiacode.nic.in" in first_cite["source_url"]


# ── 4. Empty Retrieval Results Tests ─────────────────────────────────────────

class TestEmptyRetrievalResults:
    """Test handling when knowledge base has 0 indexed provisions."""

    def test_empty_corpus_guidance(self):
        synthesizer = LegalSynthesizer()
        result = synthesizer.synthesize_with_metadata(
            query="grounds for divorce",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=True,
        )

        assert result.evidence_status == "empty_corpus"
        assert "Knowledge Base Not Yet Indexed" in result.reply
        assert "python -m app.ingestion.pipeline" in result.reply


# ── 5. LLM Provider Failure Fallback Tests ───────────────────────────────────

class TestLLMProviderFailures:
    """Ensure external network or API provider errors never crash the application."""

    def test_llm_network_timeout_fallback(self):
        llm = LLMService(provider="openai", api_key="sk-mock-key")
        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Connection timed out")):
            out = llm.generate("Test prompt")
            assert out is None  # Graceful fallback, no exception raised

    def test_llm_http_500_error_fallback(self):
        llm = LLMService(provider="ollama")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error in Ollama daemon"

        with patch("httpx.Client.post", return_value=mock_resp):
            out = llm.generate("Test prompt")
            assert out is None


# ── 6. Safety & Immediate Danger Routing Tests ───────────────────────────────

class TestSafetyAndEmergencyRouting:
    """Verify that acute physical danger queries prioritize verified government helplines."""

    def test_immediate_physical_danger_triggers_emergency_status(self):
        auditor = SafetyAuditor()
        res = auditor.screen_query("He is beating me and threatening to kill me right now please help")
        assert res.is_emergency is True
        assert "112" in (res.response_text or "")
        assert "181" in (res.response_text or "")
        assert "1091" in (res.response_text or "")

    def test_hindi_emergency_screening(self):
        auditor = SafetyAuditor()
        res = auditor.screen_query("मुझे बचाओ वह मारपीट कर रहा है जान का खतरा है", language=SupportedLanguage.HINDI)
        assert res.is_emergency is True
        assert "112" in (res.response_text or "")
        assert "181" in (res.response_text or "")

    def test_agent_routes_emergency_directly(self):
        agent = LegalAgent()
        req = ChatRequest(
            message="Someone is physically assaulting me right now emergency",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)

        assert resp.evidence_status == "emergency"
        assert "112" in resp.reply
        assert "181" in resp.reply
        assert resp.agent_steps[0]["name"] == "Immediate Safety Screening"


# ── 7. API Response Validation Tests ─────────────────────────────────────────

class TestApiResponseValidation:
    """Verify /api/chat endpoint conforms to complete schema with Step 4 metadata."""

    def test_chat_endpoint_contains_step4_fields(self, client):
        response = client.post(
            "/api/chat",
            json={
                "message": "What are the legal provisions related to maintenance?",
                "language": "en",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "evidence_status" in data
        assert "citations" in data
        assert "warnings" in data
        assert "retrieved_passages" in data
        assert "agent_steps" in data
        assert "disclaimer" in data
        assert isinstance(data["citations"], list)
        assert isinstance(data["warnings"], list)
