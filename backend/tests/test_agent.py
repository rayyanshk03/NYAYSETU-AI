"""Tests for the Step 4 agentic AI layer.

Tests cover:
- QueryPlanner intent classification for all supported intents
- QueryPlanner sub-query generation
- LegalSynthesizer response for empty corpus, insufficient evidence, grounded answer
- LegalAgent orchestrator with mocked retriever (empty corpus path)
- /api/chat endpoint integration with the new agent
- /api/retrieve endpoint
- /api/corpus/status endpoint
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.agent.planner import QueryPlanner, QueryIntent, SupportedLanguage
from app.agent.synthesizer import LegalSynthesizer
from app.agent.orchestrator import LegalAgent
from app.schemas.chat import ChatRequest, SupportedLanguage as ChatLanguage
from app.schemas.retrieval import RetrievedPassage


# ── Fixtures ─────────────────────────────────────────────────────────────────

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
    document_title: str = "Hindu Marriage Act, 1955",
    act_name: str = "Hindu Marriage Act, 1955",
    section_number: str = "13",
    section_title: str = "Divorce",
    legal_text: str = "Either party to a marriage may present a petition for dissolution.",
    score: float = 0.75,
    source_url: str = "https://www.indiacode.nic.in/handle/123456789/2007",
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        act_name=act_name,
        section_number=section_number,
        section_title=section_title,
        page_start=1,
        page_end=1,
        source_url=source_url,
        legal_text=legal_text,
        score=score,
    )


# ── QueryPlanner Tests ────────────────────────────────────────────────────────

class TestQueryPlanner:
    @pytest.mark.parametrize(
        "query, expected_intent",
        [
            ("What are the grounds for divorce under the Hindu Marriage Act?", QueryIntent.DIVORCE),
            ("How much alimony can a wife claim after filing for maintenance?", QueryIntent.MAINTENANCE),
            ("Who gets custody of a minor child and what are parental rights?", QueryIntent.CUSTODY),
            ("What protection orders are available under the domestic violence act?", QueryIntent.DOMESTIC_VIOLENCE),
            ("How can I adopt a child in India?", QueryIntent.ADOPTION),
            ("What are the conditions for a valid marriage under the Special Marriage Act?", QueryIntent.MARRIAGE),
        ],
    )
    def test_intent_classification(self, planner, query, expected_intent):
        plan = planner.plan(query, SupportedLanguage.ENGLISH)
        assert plan.intent == expected_intent
        assert plan.confidence > 0.0

    def test_generates_sub_queries(self, planner):
        plan = planner.plan("grounds for divorce", SupportedLanguage.ENGLISH)
        assert len(plan.sub_queries) >= 1
        for sq in plan.sub_queries:
            assert sq.text
            assert sq.top_k >= 1

    def test_plan_contains_original_query(self, planner):
        query = "What is alimony?"
        plan = planner.plan(query, SupportedLanguage.ENGLISH)
        assert plan.original_query == query

    def test_hindi_query_classified(self, planner):
        plan = planner.plan("talak ke kya niyam hain", SupportedLanguage.HINDI)
        assert plan.intent == QueryIntent.DIVORCE

    def test_general_intent_with_clarification(self, planner):
        plan = planner.plan("tell me about laws", SupportedLanguage.ENGLISH)
        # May or may not trigger clarification, but should not error
        assert plan.intent is not None
        assert plan.language == SupportedLanguage.ENGLISH

    def test_empty_query_defaults_to_general(self, planner):
        # Very short / nonsense query
        plan = planner.plan("???", SupportedLanguage.ENGLISH)
        assert plan.intent == QueryIntent.GENERAL


# ── LegalSynthesizer Tests ────────────────────────────────────────────────────

class TestLegalSynthesizer:
    def test_empty_corpus_response_english(self, synthesizer):
        result = synthesizer.synthesize(
            query="divorce grounds",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=True,
        )
        assert "Knowledge Base Not Yet Indexed" in result or "empty" in result.lower()

    def test_empty_corpus_response_hindi(self, synthesizer):
        result = synthesizer.synthesize(
            query="talak",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.HINDI,
            passages=[],
            corpus_empty=True,
        )
        assert "ज्ञान आधार" in result or "NyayaSetu" in result or "indexed" in result.lower()

    def test_insufficient_evidence_response(self, synthesizer):
        result = synthesizer.synthesize(
            query="extremely niche obscure query xyz",
            intent=QueryIntent.GENERAL,
            language=SupportedLanguage.ENGLISH,
            passages=[],
            corpus_empty=False,
        )
        assert "Insufficient Evidence" in result or "insufficient" in result.lower()

    def test_grounded_response_contains_section(self, synthesizer):
        p = _make_passage()
        result = synthesizer.synthesize(
            query="divorce under Hindu Marriage Act",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=[p],
            corpus_empty=False,
        )
        assert "Hindu Marriage Act" in result
        assert "§ 13" in result or "Divorce" in result
        assert "⚠️" in result  # Disclaimer present

    def test_low_score_passage_not_cited(self, synthesizer):
        p = _make_passage(score=0.10)  # Below _MIN_CITE_SCORE of 0.25
        result = synthesizer.synthesize(
            query="divorce",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=[p],
            corpus_empty=False,
        )
        assert "Insufficient Evidence" in result or "insufficient" in result.lower()

    def test_disclaimer_always_present(self, synthesizer):
        p = _make_passage(score=0.80)
        result = synthesizer.synthesize(
            query="divorce",
            intent=QueryIntent.DIVORCE,
            language=SupportedLanguage.ENGLISH,
            passages=[p],
            corpus_empty=False,
        )
        assert "not legal advice" in result.lower() or "⚠️" in result


# ── LegalAgent Orchestrator Tests ─────────────────────────────────────────────

class TestLegalAgent:
    def _make_agent_with_empty_corpus(self) -> LegalAgent:
        """Return an agent whose vector store reports 0 documents."""
        mock_vector_store = MagicMock()
        mock_vector_store.count.return_value = 0

        mock_retriever = MagicMock()
        mock_retriever.vector_store = mock_vector_store

        return LegalAgent(retriever=mock_retriever)

    def test_agent_handles_empty_corpus(self):
        ag = self._make_agent_with_empty_corpus()
        req = ChatRequest(message="What are divorce grounds?", language=ChatLanguage.ENGLISH)
        response = ag.run(req)
        assert response.status == "success"
        assert response.is_placeholder is False
        assert len(response.retrieved_passages) == 0
        assert response.reply  # Some text must be returned
        assert len(response.agent_steps) >= 2  # Plan + retrieve steps at minimum

    def test_agent_returns_correct_message_echo(self):
        ag = self._make_agent_with_empty_corpus()
        msg = "How do I apply for maintenance?"
        req = ChatRequest(message=msg, language=ChatLanguage.ENGLISH)
        response = ag.run(req)
        assert response.received_message == msg

    def test_agent_steps_have_required_fields(self):
        ag = self._make_agent_with_empty_corpus()
        req = ChatRequest(message="custody rights", language=ChatLanguage.ENGLISH)
        response = ag.run(req)
        for step in response.agent_steps:
            assert "step" in step
            assert "name" in step
            assert "description" in step
            assert "result_summary" in step

    def test_agent_language_preserved(self):
        ag = self._make_agent_with_empty_corpus()
        req = ChatRequest(message="talak ke niyam", language=ChatLanguage.HINDI)
        response = ag.run(req)
        assert response.language == ChatLanguage.HINDI


# ── API Endpoint Integration Tests ───────────────────────────────────────────

class TestAgentEndpoints:
    def test_chat_returns_200(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": "What are divorce grounds?", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("success", "error")
        assert "reply" in data
        assert "retrieved_passages" in data
        assert "agent_steps" in data

    def test_chat_hindi_language(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": "talak ke kya niyam hain", "language": "hi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "hi"

    def test_chat_empty_message_rejected(self, client):
        resp = client.post(
            "/api/chat",
            json={"message": "   ", "language": "en"},
        )
        assert resp.status_code == 422

    def test_retrieve_endpoint_empty_corpus(self, client):
        resp = client.post(
            "/api/retrieve",
            json={"query": "divorce grounds Hindu Marriage Act", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "passages" in data
        assert "total_results" in data
        assert isinstance(data["passages"], list)

    def test_corpus_status_endpoint(self, client):
        resp = client.get("/api/corpus/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "indexed_provisions" in data
        assert "status" in data
        assert data["status"] in ("ready", "empty")

    def test_health_still_works(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
