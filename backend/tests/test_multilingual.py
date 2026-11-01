"""Tests for Multilingual Legal Understanding (Prompt 6).

Covers:
1. English, Hindi (Devanagari & transliterated), and Kannada (Kannada script & transliterated) queries
2. Explicit language selection overriding automatic detection
3. Uncertain language detection triggering a user selection prompt
4. Preservation of official statutory Act names, section numbers, dates, and India Code citations
5. Plain-language explanation vs official translation clarifications
6. Multilingual semantic retrieval against synthetic test passages
7. Empty corpus abstention behavior in all three languages
8. Citation metadata preservation across languages
"""

import pytest
from unittest.mock import MagicMock

from app.schemas.chat import ChatRequest, SupportedLanguage
from app.schemas.retrieval import RetrievedPassage
from app.agent.language_detector import language_detector, LanguageDetector
from app.agent.planner import QueryPlanner, QueryIntent, UserIntentType
from app.agent.synthesizer import LegalSynthesizer
from app.agent.orchestrator import LegalAgent


# ── Synthetic Test Fixtures ──────────────────────────────────────────────────

def _make_synthetic_passage(
    chunk_id: str = "synth_hma_13b_01",
    document_id: str = "hma_1955",
    document_title: str = "The Hindu Marriage Act, 1955",
    act_name: str = "Hindu Marriage Act, 1955",
    section_number: str = "13B",
    section_title: str = "Divorce by mutual consent",
    score: float = 0.88,
) -> RetrievedPassage:
    """Create synthetic passage for test fixtures; not real legal evidence."""
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        act_name=act_name,
        section_number=section_number,
        section_title=section_title,
        subsection_id=None,
        page_start=12,
        page_end=13,
        page_number=12,
        source_url="https://www.indiacode.nic.in/handle/123456789/1560",
        publication_date="1955-05-18",
        effective_date="1955-05-18",
        document_version="1955",
        legal_text=(
            "[SYNTHETIC TEST PASSAGE] Section 13B: Divorce by mutual consent. "
            "Both parties may jointly petition the Family Court for dissolution "
            "of marriage on grounds of living separately for one year."
        ),
        score=score,
        is_split_fallback=False,
        metadata={"test_fixture": True},
    )


def _agent_with_passages(passages: list[RetrievedPassage], corpus_count: int = 10) -> LegalAgent:
    """Return a LegalAgent with mocked retriever and vector store for unit tests."""
    mock_retriever = MagicMock()
    mock_retriever.vector_store.count.return_value = corpus_count

    mock_resp = MagicMock()
    mock_resp.passages = passages
    mock_resp.total_results = len(passages)
    mock_resp.insufficient_evidence = len(passages) == 0
    mock_retriever.retrieve.return_value = mock_resp

    return LegalAgent(retriever=mock_retriever)


# ── 1. Language Detection & Resolution Tests ─────────────────────────────────

class TestLanguageDetectionAndResolution:
    """Verify script and token-based language detection and resolution priority."""

    def test_detect_english_script_and_tokens(self):
        result = language_detector.detect_language(
            "What are the statutory grounds for divorce under the Hindu Marriage Act?"
        )
        assert result.detected_language == SupportedLanguage.ENGLISH
        assert result.confidence >= 0.70
        assert not result.is_uncertain
        assert "english" in result.method or "latin" in result.method

    def test_detect_hindi_devanagari_script(self):
        result = language_detector.detect_language(
            "हिंदू विवाह अधिनियम के तहत तलाक के आधार क्या हैं?"
        )
        assert result.detected_language == SupportedLanguage.HINDI
        assert result.confidence >= 0.90
        assert not result.is_uncertain
        assert result.method == "script_devanagari"

    def test_detect_kannada_script(self):
        result = language_detector.detect_language(
            "ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆಯಡಿ ವಿಚ್ಛೇದನಕ್ಕೆ ಯಾವ ಆಧಾರಗಳಿವೆ?"
        )
        assert result.detected_language == SupportedLanguage.KANNADA
        assert result.confidence >= 0.90
        assert not result.is_uncertain
        assert result.method == "script_kannada"

    def test_detect_transliterated_hindi(self):
        result = language_detector.detect_language(
            "talak ke niyam kya hain aur patni ka bharan poshan kaise hoga?"
        )
        assert result.detected_language == SupportedLanguage.HINDI
        assert result.confidence >= 0.70
        assert not result.is_uncertain
        assert result.method == "latin_transliterated_hindi"

    def test_detect_transliterated_kannada(self):
        result = language_detector.detect_language(
            "vivaha vichhedanege yava kanonigaive mathu jeevanamsha hege padeyuvudu?"
        )
        assert result.detected_language == SupportedLanguage.KANNADA
        assert result.confidence >= 0.70
        assert not result.is_uncertain
        assert result.method == "latin_transliterated_kannada"

    def test_detect_uncertain_numbers_or_symbols(self):
        """Numbers, symbols, or unidentifiable gibberish must be marked as uncertain."""
        result = language_detector.detect_language("123456 ??? @@##")
        assert result.is_uncertain
        assert result.detected_language is None

    def test_detect_uncertain_empty_input(self):
        result = language_detector.detect_language("   ")
        assert result.is_uncertain
        assert result.detected_language is None

    def test_explicit_selection_overrides_detection(self):
        """Explicit user language takes priority even if query script differs."""
        # User typed English text, but explicitly selected Hindi
        res = language_detector.resolve_language(
            query="What are the grounds for divorce?",
            explicit_language=SupportedLanguage.HINDI,
        )
        assert res.language == SupportedLanguage.HINDI
        assert res.is_explicit is True
        assert res.is_uncertain is False

    def test_auto_triggers_detection(self):
        """Passing explicit_language=None or AUTO runs automatic detection."""
        res = language_detector.resolve_language(
            query="सीआरपीसी धारा 125 के तहत भरण पोषण",
            explicit_language=None,
        )
        assert res.language == SupportedLanguage.HINDI
        assert res.is_explicit is False
        assert res.is_uncertain is False

    def test_uncertain_detection_in_agent_prompts_user(self):
        """When detection is uncertain and no language was chosen, orchestrator prompts the user."""
        agent = LegalAgent()
        req = ChatRequest(message="12345 67890 ???", language=None)
        resp = agent.run(req)

        assert resp.status == "clarification_required"
        assert resp.evidence_status == "language_selection"
        assert resp.language_uncertain is True
        assert "Language Selection Required" in resp.reply
        assert "English" in resp.reply
        assert "हिंदी" in resp.reply
        assert "ಕನ್ನಡ" in resp.reply


# ── 2. Multilingual Intent Classification ────────────────────────────────────

class TestMultilingualIntentPlanning:
    """Verify intent planner correctly identifies topics across all three languages."""

    @pytest.fixture
    def planner(self):
        return QueryPlanner()

    def test_english_divorce_planning(self, planner):
        plan = planner.plan(
            "What are the statutory grounds for divorce under the Hindu Marriage Act?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent == QueryIntent.DIVORCE
        assert plan.topic == "Divorce"
        assert plan.language == SupportedLanguage.ENGLISH

    def test_hindi_devanagari_divorce_planning(self, planner):
        plan = planner.plan(
            "हिंदू विवाह अधिनियम के तहत तलाक के आधार क्या हैं?",
            SupportedLanguage.HINDI,
        )
        assert plan.intent == QueryIntent.DIVORCE
        assert plan.confidence >= 0.3
        assert plan.language == SupportedLanguage.HINDI

    def test_kannada_script_divorce_planning(self, planner):
        plan = planner.plan(
            "ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆಯಡಿ ವಿಚ್ಛೇದನಕ್ಕೆ ಯಾವ ಆಧಾರಗಳಿವೆ?",
            SupportedLanguage.KANNADA,
        )
        assert plan.intent == QueryIntent.DIVORCE
        assert plan.confidence >= 0.3
        assert plan.language == SupportedLanguage.KANNADA

    def test_hindi_devanagari_maintenance_planning(self, planner):
        plan = planner.plan(
            "सीआरपीसी धारा 125 के तहत पत्नी का भरण-पोषण कैसे निर्धारित होता है?",
            SupportedLanguage.HINDI,
        )
        assert plan.intent == QueryIntent.MAINTENANCE
        assert plan.language == SupportedLanguage.HINDI

    def test_kannada_script_maintenance_planning(self, planner):
        plan = planner.plan(
            "ಸೆಕ್ಷನ್ 125 ರ ಪ್ರಕಾರ ಹೆಂಡತಿಗೆ ಮಧ್ಯಂತರ ಜೀವನಾಂಶ ಪಡೆಯುವ ಪ್ರಕ್ರಿಯೆ ಏನು?",
            SupportedLanguage.KANNADA,
        )
        assert plan.intent == QueryIntent.MAINTENANCE
        assert plan.language == SupportedLanguage.KANNADA

    def test_hindi_custody_planning(self, planner):
        plan = planner.plan(
            "कस्टडी विवाद में बच्चे के सर्वोत्तम कल्याण का क्या अर्थ है?",
            SupportedLanguage.HINDI,
        )
        assert plan.intent == QueryIntent.CUSTODY
        assert plan.language == SupportedLanguage.HINDI

    def test_kannada_domestic_violence_planning(self, planner):
        plan = planner.plan(
            "ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ ತಡೆ ಕಾಯ್ದೆಯಡಿ ರಕ್ಷಣಾ ಆದೇಶ ಹೇಗೆ ಪಡೆಯಬಹುದು?",
            SupportedLanguage.KANNADA,
        )
        assert plan.intent == QueryIntent.DOMESTIC_VIOLENCE
        assert plan.language == SupportedLanguage.KANNADA

    def test_planner_retains_original_query_in_subqueries(self, planner):
        query = "ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆಯಡಿ ವಿಚ್ಛೇದನಕ್ಕೆ ಯಾವ ಆಧಾರಗಳಿವೆ?"
        plan = planner.plan(query, SupportedLanguage.KANNADA)
        assert plan.sub_queries[0].text == query


# ── 3. Legal Meaning & Terminology Preservation Tests ────────────────────────

class TestLegalMeaningPreservation:
    """Verify Act names, section numbers, disclaimers, and interpretation warnings."""

    def test_english_grounded_response_preserves_statute_names(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        req = ChatRequest(
            message="What are divorce grounds under HMA?",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)

        assert resp.language == SupportedLanguage.ENGLISH
        assert "Hindu Marriage Act, 1955" in resp.reply
        assert "Section 13B" in resp.reply
        assert "Disclaimer:" in resp.reply
        assert "Plain-Language Interpretation" in resp.reply

    def test_hindi_grounded_response_preserves_statute_names_and_section(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        req = ChatRequest(
            message="हिंदू विवाह अधिनियम के तहत तलाक के आधार क्या हैं?",
            language=SupportedLanguage.HINDI,
        )
        resp = agent.run(req)

        assert resp.language == SupportedLanguage.HINDI
        # Official statute name and section must be intact
        assert "Hindu Marriage Act, 1955" in resp.reply
        assert "Section 13B" in resp.reply
        # Hindi interpretation heading and plain-language clarification note must be present
        assert "सरल भाषा में विधिक व्याख्या" in resp.reply
        assert "आधिकारिक अनुवाद नहीं" in resp.reply
        # Hindi disclaimer must be present
        assert "अस्वीकरण" in resp.reply or "Disclaimer" in resp.reply

    def test_kannada_grounded_response_preserves_statute_names_and_section(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        req = ChatRequest(
            message="ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆಯಡಿ ವಿಚ್ಛೇದನಕ್ಕೆ ಯಾವ ಆಧಾರಗಳಿವೆ?",
            language=SupportedLanguage.KANNADA,
        )
        resp = agent.run(req)

        assert resp.language == SupportedLanguage.KANNADA
        # Official statute name and section must be intact
        assert "Hindu Marriage Act, 1955" in resp.reply
        assert "Section 13B" in resp.reply
        # Kannada interpretation heading and plain-language clarification note must be present
        assert "ಸರಳ ಭಾಷೆಯ ಕಾನೂನು ವ್ಯಾಖ್ಯಾನ" in resp.reply
        assert "ಅಧಿಕೃತ ಕಾನೂನು ಅನುವಾದವಲ್ಲ" in resp.reply
        # Kannada disclaimer must be present
        assert "ಹಕ್ಕುತ್ಯಾಗ" in resp.reply or "Disclaimer" in resp.reply

    def test_citations_metadata_contains_unmodified_statute_name(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        req = ChatRequest(
            message="ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆ",
            language=SupportedLanguage.KANNADA,
        )
        resp = agent.run(req)

        assert len(resp.citations) == 1
        citation = resp.citations[0]
        assert citation["act_name"] == "Hindu Marriage Act, 1955"
        assert citation["section_number"] == "13B"
        assert "indiacode.nic.in" in citation["source_url"]


# ── 4. Empty Corpus Abstention in All Three Languages ────────────────────────

class TestEmptyCorpusMultilingualAbstention:
    """Verify empty corpus behavior gracefully returns localized guidance."""

    def test_empty_corpus_english(self):
        synthesizer = LegalSynthesizer()
        res = synthesizer.synthesize_with_metadata(
            query="What is divorce?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=True,
        )
        assert res.evidence_status == "empty_corpus"
        assert "NyayaSetu AI" in res.reply
        assert "indexed" in res.reply.lower()
        assert len(res.citations) == 0

    def test_empty_corpus_hindi(self):
        synthesizer = LegalSynthesizer()
        res = synthesizer.synthesize_with_metadata(
            query="तलाक क्या है?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.HINDI,
            passages=[],
            corpus_empty=True,
        )
        assert res.evidence_status == "empty_corpus"
        assert "NyayaSetu AI" in res.reply
        assert "ज्ञान आधार" in res.reply
        assert len(res.citations) == 0

    def test_empty_corpus_kannada(self):
        synthesizer = LegalSynthesizer()
        res = synthesizer.synthesize_with_metadata(
            query="ವಿಚ್ಛೇದನ ಎಂದರೇನು?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.KANNADA,
            passages=[],
            corpus_empty=True,
        )
        assert res.evidence_status == "empty_corpus"
        assert "NyayaSetu AI" in res.reply
        assert "ಜ್ಞಾನ ಆಧಾರ" in res.reply
        assert len(res.citations) == 0


# ── 5. End-to-End Multilingual Chat Flow ─────────────────────────────────────

class TestEndToEndMultilingualChat:
    """Verify full agent execution preserves selected language end-to-end."""

    def test_auto_detect_hindi_end_to_end(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        # Client did not specify language; should auto-detect Hindi
        req = ChatRequest(message="हिंदू विवाह अधिनियम के तहत तलाक के आधार क्या हैं?", language=None)
        resp = agent.run(req)

        assert resp.language == SupportedLanguage.HINDI
        assert resp.detected_language == SupportedLanguage.HINDI
        assert resp.language_detection_method == "script_devanagari"
        assert resp.language_uncertain is False
        assert "Hindu Marriage Act, 1955" in resp.reply

    def test_auto_detect_kannada_end_to_end(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        # Client did not specify language; should auto-detect Kannada
        req = ChatRequest(message="ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆಯಡಿ ವಿಚ್ಛೇದನಕ್ಕೆ ಯಾವ ಆಧಾರಗಳಿವೆ?", language=None)
        resp = agent.run(req)

        assert resp.language == SupportedLanguage.KANNADA
        assert resp.detected_language == SupportedLanguage.KANNADA
        assert resp.language_detection_method == "script_kannada"
        assert resp.language_uncertain is False

    def test_explicit_language_overrides_detected_script(self):
        passage = _make_synthetic_passage()
        agent = _agent_with_passages([passage])
        # User typed Devanagari Hindi text, but explicitly selected English
        req = ChatRequest(
            message="हिंदू विवाह अधिनियम के तहत तलाक",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)

        # Output must be generated in the user-selected language (English)
        assert resp.language == SupportedLanguage.ENGLISH
        assert "Legal Overview" in resp.reply or "Statutory Provisions" in resp.reply
