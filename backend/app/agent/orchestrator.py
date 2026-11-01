"""Agentic orchestrator for NyayaSetu AI.

Connects the emergency safety screen → planner → retriever → synthesizer pipeline
and returns a structured ChatResponse with grounded statutory citations and
an explainable agent execution trace.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from app.agent.planner import QueryPlanner, QueryPlan, planner as default_planner
from app.agent.synthesizer import LegalSynthesizer, SynthesizedResult, synthesizer as default_synthesizer
from app.agent.safety import SafetyAuditor, safety_auditor as default_safety_auditor
from app.agent.language_detector import LanguageDetector, language_detector as default_language_detector
from app.rag.retrieval import LegalRetriever, retrieval_service as default_retriever
from app.rag.hybrid_retriever import HybridRetriever, hybrid_retrieval_service as default_hybrid_retriever
from app.schemas.chat import ChatRequest, ChatResponse, SupportedLanguage
from app.schemas.retrieval import RetrievedPassage

logger = logging.getLogger(__name__)


class AgentStep(BaseModel):
    """A single recorded step in the agent's execution trace."""

    step: int
    name: str
    description: str
    result_summary: str
    duration_ms: Optional[float] = None


class LegalAgent:
    """
    Agentic orchestrator for Indian family law queries.

    Execution flow:
    1. Language Resolution — explicit user choice takes priority; auto-detects otherwise; prompts user if uncertain
    2. Safety Check — screen for immediate physical danger / crisis; route to 112/181 if found
    3. Plan — classify intent and build retrieval sub-queries
    4. Retrieve — hybrid search (dense ChromaDB + sparse BM25 + RRF + cross-encoder rerank)
       - Per-sub-query exception isolation: one failure does not abort the pipeline
       - Query-text deduplication: same text is not sent to the retriever twice
    5. Deduplicate & rank passages by chunk_id
    6. Synthesize — build grounded, citation-validated answer via LLM or template
    7. Assemble ChatResponse with evidence status, clarification_question, topic,
       citations, warnings, language detection metadata, and execution trace
    """

    def __init__(
        self,
        planner: Optional[QueryPlanner] = None,
        retriever: Optional[Any] = None,
        synthesizer: Optional[LegalSynthesizer] = None,
        safety: Optional[SafetyAuditor] = None,
        language_detector: Optional[LanguageDetector] = None,
    ):
        self.planner = planner or default_planner
        self.retriever = retriever or default_hybrid_retriever
        self.synthesizer = synthesizer or default_synthesizer
        self.safety = safety or default_safety_auditor
        self.language_detector = language_detector or default_language_detector

    def run(self, request: ChatRequest) -> ChatResponse:
        """
        Execute the full agentic pipeline for a user legal query.

        Args:
            request: Validated ChatRequest with message and language.
        Returns:
            ChatResponse with grounded reply, retrieved passages, citations, and agent trace.
        """
        steps: List[AgentStep] = []
        step_idx = 0
        all_passages: List[RetrievedPassage] = []
        corpus_empty = False

        try:
            # ── Language Resolution & Detection ───────────────────────────
            # Rule 1: Explicit user-selected language ALWAYS overrides automatic detection.
            # Rule 2: If language is unspecified or 'auto', detect English, Hindi, or Kannada.
            # Rule 3: If detection is uncertain, prompt the user to select rather than guessing.
            req_lang = (
                request.language
                if request.language is not None and request.language != SupportedLanguage.AUTO
                else None
            )
            resolution = self.language_detector.resolve_language(
                query=request.message,
                explicit_language=req_lang,
            )
            effective_language = resolution.language
            detection = resolution.detection

            # ── Step 1: Immediate Safety Screening (FIRST RECORDED TRACE STEP) ─
            step_idx += 1
            t0 = _now_ms()
            emergency_check = self.safety.screen_query(request.message, effective_language)
            safety_ms = _now_ms() - t0

            if emergency_check.is_emergency:
                steps.append(
                    AgentStep(
                        step=step_idx,
                        name="Immediate Safety Screening",
                        description="Screened input for imminent physical danger or crisis distress.",
                        result_summary="IMMEDIATE DANGER DETECTED — Prioritized official national emergency resources (112, 181, 1091).",
                        duration_ms=safety_ms,
                    )
                )
                return ChatResponse(
                    status="success",
                    message_id=str(uuid.uuid4()),
                    received_message=request.message,
                    language=effective_language,
                    reply=emergency_check.response_text or "",
                    is_placeholder=False,
                    evidence_status="emergency",
                    citations=[],
                    warnings=["User query matched imminent danger criteria; official emergency helpline resources were prioritized."],
                    timestamp=datetime.now(timezone.utc),
                    retrieved_passages=[],
                    agent_steps=[s.model_dump() for s in steps],
                    detected_language=detection.detected_language,
                    language_detection_method=detection.method,
                    language_uncertain=False,
                )

            steps.append(
                AgentStep(
                    step=step_idx,
                    name="Immediate Safety Screening",
                    description="Screened input for imminent physical danger or crisis distress.",
                    result_summary="Passed safety screen; proceeding to statutory legal decision support.",
                    duration_ms=safety_ms,
                )
            )

            # Handle uncertain language detection when user made no explicit choice
            if resolution.is_uncertain:
                step_idx += 1
                steps.append(
                    AgentStep(
                        step=step_idx,
                        name="Language Resolution",
                        description="Analyzed input script and vocabulary to resolve communication language.",
                        result_summary="Language detection uncertain; requesting explicit language selection from user.",
                        duration_ms=0,
                    )
                )
                return ChatResponse(
                    status="clarification_required",
                    message_id=str(uuid.uuid4()),
                    received_message=request.message,
                    language=SupportedLanguage.ENGLISH,
                    reply=(
                        "**[Language Selection Required / भाषा चयन आवश्यक / ಭಾಷೆ ಆಯ್ಕೆ ಅಗತ್ಯ]**\n\n"
                        "We could not reliably determine your preferred language for this query. "
                        "Please select your language to proceed:\n\n"
                        "• **English**: Continue in English\n"
                        "• **हिंदी (Hindi)**: हिंदी में कानूनी जानकारी के लिए भाषा चुनें\n"
                        "• **ಕನ್ನಡ (Kannada)**: ಕನ್ನಡದಲ್ಲಿ ಕಾನೂನು ಮಾಹಿತಿಗಾಗಿ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ\n\n"
                        "*(NyayaSetu AI supports English, Hindi, and Kannada for Indian family law decision support.)*"
                    ),
                    is_placeholder=False,
                    evidence_status="language_selection",
                    citations=[],
                    warnings=["Query language could not be determined with sufficient confidence. User language selection requested."],
                    timestamp=datetime.now(timezone.utc),
                    retrieved_passages=[],
                    agent_steps=[s.model_dump() for s in steps],
                    clarification_question="Please select your preferred language: English, Hindi, or Kannada.",
                    topic="Language Selection",
                    detected_language=None,
                    language_detection_method="uncertain",
                    language_uncertain=True,
                )

            step_idx += 1
            steps.append(
                AgentStep(
                    step=step_idx,
                    name="Language Resolution",
                    description="Resolved query language via explicit user selection or automatic script detection.",
                    result_summary=(
                        f"Resolved language: {effective_language.value.upper()} "
                        f"(explicit={resolution.is_explicit}, method={detection.method})"
                    ),
                    duration_ms=0,
                )
            )

            # ── Step 1: Plan ──────────────────────────────────────────────
            step_idx += 1
            t0 = _now_ms()
            plan: QueryPlan = self.planner.plan(request.message, effective_language)
            planning_ms = _now_ms() - t0

            steps.append(
                AgentStep(
                    step=step_idx,
                    name="Query Planning",
                    description="Classified query intent and generated targeted retrieval sub-queries.",
                    result_summary=(
                        f"Intent: {plan.intent.value} "
                        f"(confidence={plan.confidence:.0%}) | "
                        f"{len(plan.sub_queries)} sub-queries generated"
                    ),
                    duration_ms=planning_ms,
                )
            )

            # ── Step 2: Retrieve ──────────────────────────────────────────
            step_idx += 1
            t0 = _now_ms()

            # Check if corpus is empty before running retrieval
            corpus_size = self.retriever.vector_store.count()
            corpus_empty = corpus_size == 0

            seen_chunk_ids: set[str] = set()
            seen_query_texts: set[str] = set()   # Orchestrator-level dedup guard
            total_retrieved = 0
            retrieval_warnings: list[str] = []

            if not corpus_empty:
                for sq in plan.sub_queries:
                    # Skip sub-queries with identical text already sent
                    norm_text = sq.text.strip().lower()
                    if norm_text in seen_query_texts:
                        continue
                    seen_query_texts.add(norm_text)

                    try:
                        result = self.retriever.retrieve(
                            query=sq.text,
                            top_k=sq.top_k,
                            language=plan.language,
                            document_id=sq.document_hint,
                        )
                        for p in result.passages:
                            if p.chunk_id not in seen_chunk_ids:
                                seen_chunk_ids.add(p.chunk_id)
                                all_passages.append(p)
                                total_retrieved += 1
                    except Exception as retrieval_exc:
                        warn_msg = (
                            f"Retrieval sub-query failed ({type(retrieval_exc).__name__}): "
                            f"'{sq.text[:60]}'. Continuing with remaining sub-queries."
                        )
                        logger.warning(warn_msg)
                        retrieval_warnings.append(warn_msg)

            retrieve_ms = _now_ms() - t0

            # Sort deduplicated passages by score descending
            all_passages.sort(key=lambda p: p.score, reverse=True)

            steps.append(
                AgentStep(
                    step=step_idx,
                    name="Legal Retrieval",
                    description=(
                        f"Queried ChromaDB ({corpus_size} indexed provisions) "
                        f"across {len(plan.sub_queries)} unique sub-queries."
                    ),
                    result_summary=(
                        f"Retrieved {total_retrieved} unique passages "
                        f"(corpus_empty={corpus_empty})"
                        if not corpus_empty
                        else "Knowledge base is empty — indexing required before retrieval."
                    ),
                    duration_ms=retrieve_ms,
                )
            )

            # ── Step 3: Grounded Answer Synthesis ─────────────────────────
            step_idx += 1
            t0 = _now_ms()

            synth_result: SynthesizedResult = self.synthesizer.synthesize_with_metadata(
                query=request.message,
                intent=plan.intent,
                language=effective_language,
                passages=all_passages,
                clarification_prompt=plan.clarification_prompt,
                corpus_empty=corpus_empty,
            )
            synth_ms = _now_ms() - t0

            qualified_count = len(synth_result.citable_passages)
            if synth_result.evidence_status == "insufficient":
                summary_text = "No statutory provisions met the relevance threshold; returned controlled insufficient evidence response."
            elif synth_result.evidence_status == "empty_corpus":
                summary_text = "Knowledge base contains 0 indexed provisions; returned indexing guidance."
            else:
                summary_text = f"Synthesized grounded explanation citing {len(synth_result.citations)} verified statutory provision(s)."

            steps.append(
                AgentStep(
                    step=step_idx,
                    name="Grounded Answer Synthesis",
                    description="Synthesized evidence-backed explanation and validated citations against indexed provisions.",
                    result_summary=summary_text,
                    duration_ms=synth_ms,
                )
            )

            # ── Assemble response ─────────────────────────────────────────
            passages_as_dicts: List[Dict[str, Any]] = [
                _passage_to_dict(p) for p in all_passages
            ]

            # Collect retrieval warnings into synthesis warnings
            all_warnings = list(synth_result.warnings) + retrieval_warnings

            return ChatResponse(
                status="success",
                message_id=str(uuid.uuid4()),
                received_message=request.message,
                language=effective_language,
                reply=synth_result.reply,
                is_placeholder=False,
                evidence_status=synth_result.evidence_status,
                citations=synth_result.citations,
                warnings=all_warnings,
                timestamp=datetime.now(timezone.utc),
                retrieved_passages=passages_as_dicts,
                agent_steps=[s.model_dump() for s in steps],
                # Step 5: forward clarification question and legal topic
                clarification_question=plan.clarification_prompt if plan.needs_clarification else None,
                topic=plan.topic,
                # Step 6: multilingual provenance
                detected_language=detection.detected_language,
                language_detection_method=detection.method,
                language_uncertain=False,
            )

        except Exception as exc:
            logger.exception("Agent pipeline error: %s", exc)
            fallback_lang = locals().get("effective_language", SupportedLanguage.ENGLISH)
            return ChatResponse(
                status="error",
                message_id=str(uuid.uuid4()),
                received_message=request.message,
                language=fallback_lang,
                reply=(
                    "**[Internal Error]**\n\n"
                    "An unexpected error occurred while processing your legal query. "
                    "Please try again or contact support.\n\n"
                    f"*Error detail: {type(exc).__name__}: {exc}*"
                ),
                is_placeholder=False,
                evidence_status="insufficient",
                citations=[],
                warnings=[f"Pipeline error: {type(exc).__name__}"],
                timestamp=datetime.now(timezone.utc),
                retrieved_passages=[],
                agent_steps=[s.model_dump() for s in steps],
            )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _now_ms() -> float:
    """Current time in milliseconds."""
    return datetime.now(timezone.utc).timestamp() * 1000


def _passage_to_dict(p: RetrievedPassage) -> Dict[str, Any]:
    """Serialize a RetrievedPassage to a JSON-safe dict for the API response."""
    return {
        "chunk_id": p.chunk_id,
        "document_id": p.document_id,
        "document_title": p.document_title,
        "act_name": p.act_name,
        "section_number": p.section_number,
        "section_title": p.section_title,
        "subsection_id": p.subsection_id,
        "page_start": p.page_start,
        "page_end": p.page_end,
        "source_url": p.source_url,
        "publication_date": p.publication_date,
        "legal_text": p.legal_text,
        "score": p.score,
        "is_split_fallback": p.is_split_fallback,
    }


# Convenience singleton
agent = LegalAgent()
