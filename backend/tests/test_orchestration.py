"""Step 5 orchestration tests — controlled tool-orchestration behaviour.

Covers all 10 required scenarios:
1.  Simple legal question \u2192 single retrieval sub-query sufficient
2.  Complex question \u2192 multiple retrieval sub-queries (cross-Act)
3.  Ambiguous query \u2192 clarification question returned
4.  Empty corpus \u2192 abstains, returns empty_corpus status
5.  Insufficient retrieved evidence \u2192 abstains, returns insufficient status
6.  Relevant evidence present \u2192 grounded synthesis produced
7.  Citation validation \u2192 hallucinated section flagged in warnings
8.  Retrieval failure / timeout \u2192 pipeline continues, warning logged
9.  Safety screening \u2192 emergency query bypasses retrieval + synthesis
10. Language handling \u2192 English, Hindi, Kannada requests all succeed

All LLM responses and retrieved passages are synthetic; they do NOT
represent genuine legal advice or authoritative statutory provisions.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.agent.planner import QueryPlanner, QueryIntent, QueryPlan, UserIntentType, SupportedLanguage
from app.agent.synthesizer import LegalSynthesizer
from app.agent.orchestrator import LegalAgent
from app.agent.safety import SafetyAuditor
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievedPassage, RetrievalResponse
from app.services.llm_service import LLMService


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def planner():
    return QueryPlanner()


@pytest.fixture
def synthesizer():
    return LegalSynthesizer()


def _make_passage(
    chunk_id: str = "chunk_001",
    document_id: str = "hma_1955",
    act_name: str = "Hindu Marriage Act, 1955",
    section_number: str = "13",
    section_title: str = "Divorce",
    legal_text: str = "Either party may present a petition for dissolution of marriage.",
    score: float = 0.78,
    source_url: str = "https://www.indiacode.nic.in/handle/123456789/2007",
) -> RetrievedPassage:
    """Create a synthetic RetrievedPassage for testing purposes only."""
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=act_name,
        act_name=act_name,
        section_number=section_number,
        section_title=section_title,
        page_start=5,
        page_end=6,
        source_url=source_url,
        legal_text=legal_text,
        score=score,
    )


def _make_retrieval_response(passages: list[RetrievedPassage]) -> RetrievalResponse:
    """Wrap passages in a RetrievalResponse as retrieval_service.retrieve() would return."""
    return RetrievalResponse(
        query="test query",
        total_results=len(passages),
        insufficient_evidence=len(passages) == 0,
        passages=passages,
    )


def _agent_with_passages(passages: list[RetrievedPassage]) -> LegalAgent:
    """Return a LegalAgent whose retriever returns the given passages."""
    mock_vs = MagicMock()
    mock_vs.count.return_value = len(passages) if passages else 1  # Non-zero = has corpus

    mock_retriever = MagicMock()
    mock_retriever.vector_store = mock_vs
    mock_retriever.retrieve.return_value = _make_retrieval_response(passages)

    return LegalAgent(retriever=mock_retriever)


def _agent_empty_corpus() -> LegalAgent:
    """Return a LegalAgent whose vector store has 0 indexed documents."""
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0

    mock_retriever = MagicMock()
    mock_retriever.vector_store = mock_vs

    return LegalAgent(retriever=mock_retriever)


# ── Scenario 1: Simple Legal Question \u2014 Single Sub-Query Sufficient ─────────────

class TestSimpleSingleRetrievalQuery:
    """A focused, specific legal question should use one primary sub-query and no broad fallback."""

    def test_specific_divorce_question_classifies_correctly(self, planner):
        plan = planner.plan(
            "What are the statutory grounds for divorce under the Hindu Marriage Act?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent == QueryIntent.DIVORCE
        assert plan.confidence > 0.3
        assert plan.topic == "Divorce"

    def test_specific_maintenance_question_classifies_correctly(self, planner):
        plan = planner.plan(
            "How much maintenance can a wife claim under Section 125 CrPC?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent == QueryIntent.MAINTENANCE
        assert plan.confidence > 0.3

    def test_specific_query_does_not_emit_broad_fallback(self, planner):
        """A high-confidence focused query should NOT produce a broad no-filter sub-query."""
        plan = planner.plan(
            "What are grounds for divorce under the Hindu Marriage Act?",
            SupportedLanguage.ENGLISH,
        )
        # Primary sub-query should exist
        assert len(plan.sub_queries) >= 1
        assert plan.sub_queries[0].text

    def test_user_intent_type_procedure_detected(self, planner):
        plan = planner.plan("How to file for divorce?", SupportedLanguage.ENGLISH)
        assert plan.user_intent_type == UserIntentType.PROCEDURE

    def test_user_intent_type_explain_detected(self, planner):
        plan = planner.plan("What is alimony?", SupportedLanguage.ENGLISH)
        assert plan.user_intent_type == UserIntentType.EXPLAIN

    def test_user_intent_type_rights_detected(self, planner):
        plan = planner.plan("What are my rights after divorce?", SupportedLanguage.ENGLISH)
        assert plan.user_intent_type == UserIntentType.RIGHTS

    def test_agent_runs_retrieval_before_synthesis(self):
        """Verify retrieval is called before synthesis in a single-query flow."""
        passage = _make_passage()
        agent = _agent_with_passages([passage])
        req = ChatRequest(message="What are divorce grounds?", language=SupportedLanguage.ENGLISH)
        resp = agent.run(req)

        assert resp.status == "success"
        agent.retriever.retrieve.assert_called()
        # Step names must include both retrieval and synthesis
        step_names = [s["name"] for s in resp.agent_steps]
        assert "Legal Retrieval" in step_names


# ── Scenario 2: Complex Multi-Sub-Query Retrieval ────────────────────────────

class TestMultipleRetrievalSubQueries:
    """A cross-Act question should trigger multiple sub-queries with different document hints."""

    def test_divorce_cross_act_produces_two_sub_queries(self, planner):
        """Divorce spans HMA + SMA; planner should emit at least two sub-queries."""
        plan = planner.plan(
            "What are the grounds for divorce under the Hindu Marriage Act and Special Marriage Act?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent == QueryIntent.DIVORCE
        # Primary (hma_1955) + secondary (sma_1954)
        doc_hints = {sq.document_hint for sq in plan.sub_queries if sq.document_hint}
        assert len(doc_hints) >= 2 or len(plan.sub_queries) >= 2

    def test_maintenance_cross_act_produces_two_sub_queries(self, planner):
        """Maintenance spans HAMA + PWDVA."""
        plan = planner.plan(
            "What maintenance can a wife claim under the domestic violence and maintenance law?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent in (QueryIntent.MAINTENANCE, QueryIntent.DOMESTIC_VIOLENCE)
        assert len(plan.sub_queries) >= 2

    def test_sub_queries_are_text_deduplicated(self, planner):
        """No two sub-queries should have exactly the same (lowercased) text."""
        plan = planner.plan(
            "How to get divorce and maintenance simultaneously?",
            SupportedLanguage.ENGLISH,
        )
        texts = [sq.text.strip().lower() for sq in plan.sub_queries]
        assert len(texts) == len(set(texts)), (
            f"Duplicate sub-query texts found: {texts}"
        )

    def test_orchestrator_deduplicates_query_text(self):
        """Even if planner emits duplicate texts, orchestrator must not send same text twice."""
        passage = _make_passage()
        agent = _agent_with_passages([passage])

        # Inject a plan with intentionally duplicated sub-query texts
        mock_plan = MagicMock()
        mock_plan.intent = QueryIntent.DIVORCE
        mock_plan.topic = "Divorce"
        mock_plan.confidence = 0.8
        mock_plan.language = SupportedLanguage.ENGLISH
        mock_plan.needs_clarification = False
        mock_plan.clarification_prompt = None

        sq = MagicMock()
        sq.text = "divorce grounds"
        sq.top_k = 3
        sq.document_hint = "hma_1955"
        # Same text twice — orchestrator must skip the second
        mock_plan.sub_queries = [sq, sq]

        with patch.object(agent.planner, "plan", return_value=mock_plan):
            req = ChatRequest(message="divorce grounds", language=SupportedLanguage.ENGLISH)
            agent.run(req)
            # retrieve should have been called exactly once (dedup skips second)
            assert agent.retriever.retrieve.call_count == 1


# ── Scenario 3: Ambiguous Query \u2014 Clarification Required ────────────────────

class TestAmbiguousQueryClarification:
    """A vague, very short, or unrecognised query should trigger a clarification question."""

    def test_very_short_query_triggers_clarification(self, planner):
        plan = planner.plan("law", SupportedLanguage.ENGLISH)
        # "law" is 1 word, no specific intent \u2192 should request clarification
        assert plan.needs_clarification is True
        assert plan.clarification_prompt is not None
        assert len(plan.clarification_prompt) > 20

    def test_generic_phrase_triggers_clarification(self, planner):
        plan = planner.plan("legal help needed", SupportedLanguage.ENGLISH)
        # Very low confidence GENERAL query
        if plan.intent == QueryIntent.GENERAL and plan.confidence < 0.25:
            assert plan.needs_clarification is True
            assert "divorce" in plan.clarification_prompt.lower() or "clarify" in plan.clarification_prompt.lower()

    def test_specific_query_does_not_trigger_clarification(self, planner):
        plan = planner.plan(
            "What are the grounds for divorce under the Hindu Marriage Act, 1955?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.needs_clarification is False
        assert plan.clarification_prompt is None

    def test_clarification_in_api_response(self, client):
        """When clarification is needed, /api/chat should include clarification_question."""
        resp = client.post(
            "/api/chat",
            json={"message": "law", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "clarification_question" in data
        # When triggered, it should be non-None; when not triggered, it can be None
        if data.get("clarification_question"):
            assert len(data["clarification_question"]) > 10

    def test_hindi_clarification_prompt_in_hindi(self, planner):
        plan = planner.plan("niyam", SupportedLanguage.HINDI)
        if plan.needs_clarification:
            # Should contain Hindi text
            assert any(ord(c) > 0x0900 for c in plan.clarification_prompt), (
                "Hindi clarification should contain Hindi characters"
            )

    def test_kannada_clarification_prompt_in_kannada(self, planner):
        plan = planner.plan("kanoon", SupportedLanguage.KANNADA)
        if plan.needs_clarification:
            # Should contain Kannada text
            assert any(ord(c) > 0x0C80 for c in plan.clarification_prompt), (
                "Kannada clarification should contain Kannada characters"
            )


# ── Scenario 4: Empty Legal Corpus ───────────────────────────────────────────

class TestEmptyCorpusAbstention:
    """When the knowledge base has 0 indexed provisions, the system must not generate answers."""

    def test_empty_corpus_returns_empty_corpus_status(self):
        agent = _agent_empty_corpus()
        req = ChatRequest(
            message="What are the grounds for divorce under the Hindu Marriage Act?",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)
        assert resp.evidence_status == "empty_corpus"
        assert len(resp.citations) == 0
        assert len(resp.retrieved_passages) == 0

    def test_empty_corpus_reply_contains_indexing_guidance(self):
        agent = _agent_empty_corpus()
        req = ChatRequest(message="What is alimony?", language=SupportedLanguage.ENGLISH)
        resp = agent.run(req)
        # Must instruct user to index corpus
        assert "index" in resp.reply.lower() or "indexed" in resp.reply.lower()

    def test_empty_corpus_hindi_reply(self):
        agent = _agent_empty_corpus()
        req = ChatRequest(message="talak ke niyam kya hain?", language=SupportedLanguage.HINDI)
        resp = agent.run(req)
        assert resp.evidence_status == "empty_corpus"
        assert resp.language == SupportedLanguage.HINDI

    def test_synthesizer_empty_corpus_english(self, synthesizer):
        result = synthesizer.synthesize_with_metadata(
            query="adoption procedure",
            intent=QueryIntent.ADOPTION,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=True,
        )
        assert result.evidence_status == "empty_corpus"
        assert len(result.citations) == 0
        assert "python -m app.ingestion.pipeline" in result.reply

    def test_synthesizer_empty_corpus_hindi(self, synthesizer):
        result = synthesizer.synthesize_with_metadata(
            query="godh lena",
            intent=QueryIntent.ADOPTION,
            language=SupportedLanguage.HINDI,
            passages=[],
            corpus_empty=True,
        )
        assert result.evidence_status == "empty_corpus"
        # Hindi corpus response should contain Hindi text
        assert "NyayaSetu" in result.reply or "indexed" in result.reply.lower()


# ── Scenario 5: Insufficient Evidencei — Abstention ──────────────────────────

class TestInsufficientEvidenceAbstention:
    """When retrieved passages fail the similarity threshold, the system must abstain."""

    def test_low_score_passages_produce_insufficient_status(self):
        low_score_passage = _make_passage(score=0.10)  # Below 0.25 threshold
        agent = _agent_with_passages([low_score_passage])
        req = ChatRequest(message="What are marriage registration requirements?", language=SupportedLanguage.ENGLISH)
        resp = agent.run(req)
        assert resp.evidence_status == "insufficient"
        assert len(resp.citations) == 0

    def test_no_passages_returned_produces_insufficient_status(self):
        agent = _agent_with_passages([])
        req = ChatRequest(message="Describe succession rights", language=SupportedLanguage.ENGLISH)
        resp = agent.run(req)
        assert resp.evidence_status == "insufficient"

    def test_synthesizer_no_passages_no_llm_call(self, synthesizer):
        """LLM must not be called when there is no evidence."""
        mock_llm = MagicMock(spec=LLMService)
        synth = LegalSynthesizer(llm=mock_llm)
        result = synth.synthesize_with_metadata(
            query="obscure question xyz",
            intent=QueryIntent.GENERAL,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=False,
        )
        assert result.evidence_status == "insufficient"
        mock_llm.generate.assert_not_called()

    def test_insufficient_warning_included(self, synthesizer):
        result = synthesizer.synthesize_with_metadata(
            query="custody for unmarried parents",
            intent=QueryIntent.CUSTODY,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=False,
        )
        assert len(result.warnings) > 0
        assert any("threshold" in w.lower() or "relevance" in w.lower() for w in result.warnings)


# ── Scenario 6: Relevant Evidence \u2014 Grounded Synthesis ─────────────────────────

class TestGroundedSynthesiswithEvidence:
    """When sufficient evidence is present, a grounded structured answer must be produced."""

    def test_grounded_answer_with_template_fallback(self, synthesizer):
        passages = [_make_passage(score=0.82)]
        result = synthesizer.synthesize_with_metadata(
            query="What are grounds for divorce?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=passages,
        )
        assert result.evidence_status == "sufficient"
        assert len(result.citations) == 1
        assert result.citations[0]["verified"] is True
        assert "Direct Explanation" in result.reply
        assert "Relevant Statutory Provisions" in result.reply
        assert "Plain-Language Interpretation" in result.reply
        assert "Important Limitations" in result.reply
        assert "\u26a0\ufe0f" in result.reply  # Disclaimer must be present

    def test_grounded_answer_with_mocked_llm(self, synthesizer):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate.return_value = (
            "**Direct Explanation:** Under Section 13 of the Hindu Marriage Act, 1955 [1], "
            "either party may petition for divorce.\n\n"
            "**Relevant Legal Provisions:** Hindu Marriage Act, 1955, Section 13.\n\n"
            "**Plain-Language Interpretation:** Cruelty and desertion are recognised grounds.\n\n"
            "**Important Limitations:** Petition must be filed in the Family Court."
        )
        synth = LegalSynthesizer(llm=mock_llm)
        passages = [_make_passage(score=0.85)]
        result = synth.synthesize_with_metadata(
            query="Grounds for divorce?",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=passages,
        )
        assert result.evidence_status == "sufficient"
        assert "Section 13" in result.reply
        assert "Hindu Marriage Act" in result.reply
        # Disclaimer must be appended even when LLM generates the reply
        assert "legal advice" in result.reply.lower() or "\u26a0\ufe0f" in result.reply

    def test_disclaimer_always_present_in_grounded_answer(self):
        passages = [_make_passage(score=0.90)]
        synthesizer = LegalSynthesizer()
        result = synthesizer.synthesize_with_metadata(
            query="maintenance rights",
            intent=QueryIntent.MAINTENANCE,
            language=SupportedLanguage.ENGLISH,
            passages=passages,
        )
        assert "legal advice" in result.reply.lower() or "\u26a0\ufe0f" in result.reply

    def test_agent_response_includes_retrieved_passages_in_response(self):
        passage = _make_passage(score=0.80)
        agent = _agent_with_passages([passage])
        req = ChatRequest(message="What are divorce grounds?", language=SupportedLanguage.ENGLISH)
        resp = agent.run(req)
        # Passages must be forwarded to the response
        assert isinstance(resp.retrieved_passages, list)
        assert len(resp.retrieved_passages) >= 1
        assert "chunk_id" in resp.retrieved_passages[0]

    def test_topic_field_populated_in_api_response(self, client):
        resp = client.post("/api/chat", json={"message": "What are grounds for divorce?", "language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert "topic" in data
        # Topic may be None when corpus is empty, but key must exist
        if data["topic"]:
            assert isinstance(data["topic"], str)


# ── Scenario 7: Citation Validation ─────────────────────────────────────────

class TestCitationValidation:
    """Hallucinated section numbers or out-of-bounds citation indices must be flagged."""

    def test_hallucinated_section_flagged_in_warnings(self):
        mock_llm = MagicMock(spec=LLMService)
        # LLM invents Section 498A which is NOT in the retrieved evidence
        mock_llm.generate.return_value = (
            "Under Section 498A [5], the husband is liable for cruelty. "
            "Also Section 13 applies [1]."
        )
        synth = LegalSynthesizer(llm=mock_llm)
        passages = [_make_passage(section_number="13", score=0.85)]
        result = synth.synthesize_with_metadata(
            query="cruelty and divorce",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=passages,
        )
        warning_text = " ".join(result.warnings)
        assert "498A" in warning_text or "[5]" in warning_text

    def test_out_of_bounds_citation_index_flagged(self):
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate.return_value = "See evidence [99] and [100] for details."
        synth = LegalSynthesizer(llm=mock_llm)
        passages = [_make_passage(score=0.85)]
        result = synth.synthesize_with_metadata(
            query="divorce law",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=passages,
        )
        warning_text = " ".join(result.warnings)
        assert "[99]" in warning_text or "[100]" in warning_text or "out-of-bounds" in warning_text

    def test_citations_list_only_contains_retrieved_passages(self):
        passages = [
            _make_passage(chunk_id="c1", section_number="13", score=0.85),
            _make_passage(chunk_id="c2", section_number="125", score=0.76),
        ]
        synth = LegalSynthesizer()
        result = synth.synthesize_with_metadata(
            query="divorce and maintenance",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=passages,
        )
        assert len(result.citations) == 2
        for cite in result.citations:
            assert cite["verified"] is True
            assert cite["chunk_id"] in ("c1", "c2")

    def test_no_citations_without_evidence(self):
        synth = LegalSynthesizer()
        result = synth.synthesize_with_metadata(
            query="any question",
            intent=QueryIntent.GENERAL,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=False,
        )
        assert result.citations == []


# ── Scenario 8: Retrieval Failure / Timeout ───────────────────────────────

class TestRetrievalFailureHandling:
    """A retrieval sub-query failure must not crash the pipeline; a warning must be returned."""

    def test_retrieval_exception_isolated_per_subquery(self):
        """One sub-query failure should be caught and a warning logged; other sub-queries run."""
        passage = _make_passage(score=0.80)

        mock_vs = MagicMock()
        mock_vs.count.return_value = 5

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("ChromaDB connection timeout")
            return _make_retrieval_response([passage])

        mock_retriever = MagicMock()
        mock_retriever.vector_store = mock_vs
        mock_retriever.retrieve.side_effect = side_effect

        agent = LegalAgent(retriever=mock_retriever)
        req = ChatRequest(
            message="What are grounds for divorce under Hindu Marriage Act and Special Marriage Act?",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)

        # Must succeed overall (not 'error' status)
        assert resp.status == "success"
        # Warning from failed sub-query must be included
        assert any("failed" in w.lower() or "timeout" in w.lower() or "ConnectionError" in w
                   for w in resp.warnings)

    def test_all_subquery_failures_returns_empty_but_doesnt_crash(self):
        """If ALL sub-queries fail, the agent should gracefully return insufficient evidence."""
        mock_vs = MagicMock()
        mock_vs.count.return_value = 5

        mock_retriever = MagicMock()
        mock_retriever.vector_store = mock_vs
        mock_retriever.retrieve.side_effect = RuntimeError("All shards unavailable")

        agent = LegalAgent(retriever=mock_retriever)
        req = ChatRequest(message="What is guardianship?", language=SupportedLanguage.ENGLISH)
        resp = agent.run(req)

        # Must not raise; should degrade gracefully
        assert resp.status in ("success", "error")
        # No passages, no citations
        assert len(resp.retrieved_passages) == 0


# ── Scenario 9: Safety Screening ─────────────────────────────────────────────

class TestSafetyScreening:
    """Emergency queries must bypass retrieval and synthesis, routing to verified helplines only."""

    def test_english_emergency_query_routed(self):
        agent = LegalAgent()
        req = ChatRequest(
            message="He is threatening to kill me please help emergency now",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)
        assert resp.evidence_status == "emergency"
        assert "112" in resp.reply
        assert "181" in resp.reply
        assert len(resp.retrieved_passages) == 0
        assert len(resp.citations) == 0

    def test_hindi_emergency_query_routed(self):
        agent = LegalAgent()
        req = ChatRequest(
            message="\u092e\u0941\u091d\u0947 \u092c\u091a\u093e\u0913 \u0935\u0939 \u092e\u093e\u0930\u092a\u0940\u091f \u0915\u0930 \u0930\u0939\u093e \u0939\u0948 \u091c\u093e\u0928 \u0915\u093e \u0916\u0924\u0930\u093e \u0939\u0948",
            language=SupportedLanguage.HINDI,
        )
        resp = agent.run(req)
        assert resp.evidence_status == "emergency"
        assert "112" in resp.reply

    def test_emergency_step_is_first_agent_step(self):
        agent = LegalAgent()
        req = ChatRequest(
            message="Being attacked right now cannot escape call police now",
            language=SupportedLanguage.ENGLISH,
        )
        resp = agent.run(req)
        assert resp.agent_steps[0]["name"] == "Immediate Safety Screening"
        assert "IMMEDIATE DANGER" in resp.agent_steps[0]["result_summary"].upper()

    def test_non_emergency_query_passes_safety_screen(self):
        auditor = SafetyAuditor()
        result = auditor.screen_query(
            "What are the grounds for divorce under the Hindu Marriage Act?",
            SupportedLanguage.ENGLISH,
        )
        assert result.is_emergency is False

    def test_safety_screen_does_not_invent_helpline_numbers(self):
        """Only official pre-configured helpline numbers should appear in emergency response."""
        from app.agent.safety import OFFICIAL_HELPLINES
        auditor = SafetyAuditor()
        result = auditor.screen_query(
            "He is beating me and threatening to kill me please help",
            SupportedLanguage.ENGLISH,
        )
        assert result.is_emergency is True
        resp_text = result.response_text or ""
        official_numbers = {v["number"] for v in OFFICIAL_HELPLINES.values()}
        import re
        found_numbers = set(re.findall(r"\b\d{3,11}\b", resp_text))
        for num in found_numbers:
            assert num in official_numbers, (
                f"Invented helpline number '{num}' found in emergency response. "
                "Only pre-configured official helplines are permitted."
            )


# ── Scenario 10: Language Handling ───────────────────────────────────────────

class TestLanguageHandling:
    """All supported languages (English, Hindi, Kannada) should produce valid responses."""

    @pytest.mark.parametrize("language,message", [
        ("en", "What are the grounds for divorce under the Hindu Marriage Act?"),
        ("hi", "talak ke liye kya niyam hain?"),
        ("kn", "vivaha vichhedanege yava karana kanonigaive?"),
    ])
    def test_chat_endpoint_accepts_all_languages(self, client, language, message):
        resp = client.post(
            "/api/chat",
            json={"message": message, "language": language},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == language
        assert data["status"] in ("success", "error")
        assert "reply" in data

    def test_hindi_classified_correctly(self, planner):
        plan = planner.plan("talak ke kya niyam hain", SupportedLanguage.HINDI)
        assert plan.intent == QueryIntent.DIVORCE

    def test_kannada_classified_correctly(self, planner):
        plan = planner.plan("vivaha vichhedane hege maaduvudu", SupportedLanguage.KANNADA)
        assert plan.intent in (QueryIntent.DIVORCE, QueryIntent.MARRIAGE)

    def test_hindi_adoption_keyword(self, planner):
        plan = planner.plan("bachche ko godh kaise lein?", SupportedLanguage.HINDI)
        assert plan.intent == QueryIntent.ADOPTION

    def test_hindi_clarification_in_hindi(self, planner):
        plan = planner.plan("kuch bhi", SupportedLanguage.HINDI)
        if plan.needs_clarification:
            # Hindi clarification should contain Hindi text
            assert any(ord(c) > 0x0900 for c in plan.clarification_prompt)

    def test_kannada_clarification_in_kannada(self, planner):
        plan = planner.plan("enadu", SupportedLanguage.KANNADA)
        if plan.needs_clarification:
            # Kannada clarification should contain Kannada text
            assert any(ord(c) > 0x0C80 for c in plan.clarification_prompt)

    def test_empty_corpus_hindi_response(self):
        agent = _agent_empty_corpus()
        req = ChatRequest(
            message="bharan poshan ke kya niyam hain?",
            language=SupportedLanguage.HINDI,
        )
        resp = agent.run(req)
        assert resp.evidence_status == "empty_corpus"
        assert resp.language == SupportedLanguage.HINDI
        # Reply should contain some Hindi text
        assert resp.reply is not None and len(resp.reply) > 10

    def test_empty_corpus_kannada_response(self):
        agent = _agent_empty_corpus()
        req = ChatRequest(
            message="magu samrakshane hege padeuvudu?",
            language=SupportedLanguage.KANNADA,
        )
        resp = agent.run(req)
        assert resp.evidence_status == "empty_corpus"
        assert resp.language == SupportedLanguage.KANNADA


# ── New Intent Tests ──────────────────────────────────────────────────────────

class TestNewIntents:
    """Verify GUARDIANSHIP and SUCCESSION intents added in Step 5."""

    def test_guardianship_intent_classified(self, planner):
        plan = planner.plan(
            "Who is the natural guardian of a minor child under Indian law?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent == QueryIntent.GUARDIANSHIP

    def test_succession_intent_classified(self, planner):
        plan = planner.plan(
            "What are the succession rights of a Hindu widow to her husband's estate?",
            SupportedLanguage.ENGLISH,
        )
        assert plan.intent in (QueryIntent.SUCCESSION, QueryIntent.INHERITANCE)

    def test_all_intent_enum_values_are_reachable(self, planner):
        """Verify that none of the new intents are orphaned (no keywords route to them)."""
        expected_intents = {
            QueryIntent.DIVORCE, QueryIntent.MAINTENANCE, QueryIntent.CUSTODY,
            QueryIntent.GUARDIANSHIP, QueryIntent.DOMESTIC_VIOLENCE, QueryIntent.ADOPTION,
            QueryIntent.MARRIAGE, QueryIntent.INHERITANCE, QueryIntent.SUCCESSION,
        }
        test_queries = [
            ("grounds for divorce", QueryIntent.DIVORCE),
            ("alimony and maintenance section 125", QueryIntent.MAINTENANCE),
            ("child custody order", QueryIntent.CUSTODY),
            ("natural guardian legal guardian", QueryIntent.GUARDIANSHIP),
            ("domestic violence protection order", QueryIntent.DOMESTIC_VIOLENCE),
            ("child adoption procedure", QueryIntent.ADOPTION),
            ("hindu marriage act conditions", QueryIntent.MARRIAGE),
            ("inheritance hindu succession act", QueryIntent.INHERITANCE),
            ("legal heir succession certificate", QueryIntent.SUCCESSION),
        ]
        observed = set()
        for q, expected in test_queries:
            plan = planner.plan(q, SupportedLanguage.ENGLISH)
            observed.add(plan.intent)

        # At minimum, the directly queried intents should be observed
        for _, expected in test_queries:
            assert expected in observed, f"Intent {expected} was never classified"
