"""Legal answer synthesizer for NyayaSetu AI.

Implements grounded Retrieval-Augmented Generation (RAG) answer synthesis
using retrieved statutory provisions as evidence.

Guarantees:
1. Strict Groundedness: Answers are derived exclusively from retrieved statutory passages.
2. Anti-Hallucination: Does NOT invent Act names, section numbers, court judgments, or URLs.
3. Citation Validation: Verifies that all cited sections and source IDs exist in the retrieved evidence.
4. Evidence Sufficiency: Transparently returns an insufficient-evidence response when passages do not support the query.
5. Structured Legal Format:
   - Direct Explanation
   - Relevant Statutory Provisions
   - Simple-Language Interpretation
   - Important Limitations / Conditions
   - Authoritative Citations with verified India Code URLs
6. Safety & Disclaimers: Preserves statutory disclaimers across English, Hindi, and Kannada.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.planner import QueryIntent
from app.schemas.chat import SupportedLanguage
from app.schemas.retrieval import RetrievedPassage
from app.services.llm_service import llm_service, LLMService

logger = logging.getLogger(__name__)

# Minimum similarity score for a retrieved passage to be considered qualifying evidence
_MIN_CITE_SCORE = 0.25

# Maximum passages to incorporate into the prompt context
_MAX_CITED_PASSAGES = 5


class SynthesizedResult(BaseModel):
    """Structured result returned by the LegalSynthesizer."""

    reply: str
    evidence_status: str = Field(
        default="sufficient",
        description="'sufficient', 'insufficient', 'empty_corpus', or 'emergency'",
    )
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    citable_passages: List[RetrievedPassage] = Field(default_factory=list)


class LegalSynthesizer:
    """
    Synthesizes structured, evidence-backed legal answers using configured LLM providers
    or resilient deterministic grounded synthesis.
    """

    def __init__(self, llm: Optional[LLMService] = None):
        self.llm = llm or llm_service

    def synthesize(
        self,
        query: str,
        intent: QueryIntent,
        language: SupportedLanguage,
        passages: List[RetrievedPassage],
        clarification_prompt: Optional[str] = None,
        corpus_empty: bool = False,
    ) -> str:
        """
        Legacy string-returning interface for backward compatibility.
        """
        result = self.synthesize_with_metadata(
            query=query,
            intent=intent,
            language=language,
            passages=passages,
            clarification_prompt=clarification_prompt,
            corpus_empty=corpus_empty,
        )
        return result.reply

    def synthesize_with_metadata(
        self,
        query: str,
        intent: QueryIntent,
        language: SupportedLanguage,
        passages: List[RetrievedPassage],
        clarification_prompt: Optional[str] = None,
        corpus_empty: bool = False,
    ) -> SynthesizedResult:
        """
        Execute full answer generation pipeline with citation validation and evidence checks.
        """
        # Filter to passages meeting minimum similarity threshold
        citable = [p for p in passages if p.score >= _MIN_CITE_SCORE][:_MAX_CITED_PASSAGES]

        # Case 1: Corpus is completely empty
        if corpus_empty:
            return SynthesizedResult(
                reply=self._no_corpus_response(language),
                evidence_status="empty_corpus",
                citations=[],
                warnings=["The legal knowledge base contains 0 indexed provisions. Official bare acts must be indexed."],
                citable_passages=[],
            )

        # Case 2: Insufficient evidence (no passages meeting relevance threshold)
        if not citable:
            return SynthesizedResult(
                reply=self._insufficient_evidence_response(query, language, clarification_prompt),
                evidence_status="insufficient",
                citations=[],
                warnings=["No indexed statutory provisions met the minimum relevance threshold for this query."],
                citable_passages=[],
            )

        # Build verified citations registry
        citations = self._build_citation_registry(citable)

        # Case 3: Grounded synthesis (Attempt LLM, fallback to deterministic template)
        llm_reply, warnings = self._try_llm_synthesis(query, intent, language, citable, clarification_prompt)

        if llm_reply:
            # Validate citations in LLM reply against retrieved evidence
            validated_reply, citation_warnings = self._validate_citations(llm_reply, citable)
            warnings.extend(citation_warnings)

            # Ensure mandatory legal disclaimer is present
            final_reply = self._ensure_disclaimer(validated_reply, language)
            return SynthesizedResult(
                reply=final_reply,
                evidence_status="sufficient",
                citations=citations,
                warnings=warnings,
                citable_passages=citable,
            )

        # Deterministic Grounded Template Fallback
        deterministic_reply = self._build_grounded_response(query, intent, language, citable, clarification_prompt)
        return SynthesizedResult(
            reply=deterministic_reply,
            evidence_status="sufficient",
            citations=citations,
            warnings=warnings,
            citable_passages=citable,
        )

    # ------------------------------------------------------------------
    # LLM Prompt Construction & Generation
    # ------------------------------------------------------------------

    def _try_llm_synthesis(
        self,
        query: str,
        intent: QueryIntent,
        language: SupportedLanguage,
        citable: List[RetrievedPassage],
        clarification_prompt: Optional[str],
    ) -> tuple[Optional[str], List[str]]:
        """Attempt generation using configured LLM provider."""
        warnings: List[str] = []

        system_prompt = (
            "You are NyayaSetu AI, an expert multilingual decision-support system for Indian family law.\n"
            "Your objective is to provide a clear, accessible, and accurate explanation of Indian family law "
            "strictly grounded in the provided retrieved statutory provisions.\n\n"
            "STRICT CONSTRAINTS:\n"
            "1. Answer ONLY using the provided retrieved passages. DO NOT use unverified general knowledge.\n"
            "2. DO NOT invent legal section numbers, act names, court judgments, or URLs.\n"
            "3. LEGAL MEANING PRESERVATION RULES:\n"
            "   - PRESERVE official statutory Act names (e.g. 'Hindu Marriage Act, 1955', 'Protection of Women from Domestic Violence Act, 2005'), section numbers (e.g. 'Section 13B', 'Section 125'), and dates in their standard recognized form.\n"
            "   - You may include common localized names alongside official titles (e.g. 'हिंदू विवाह अधिनियम, 1955 (Hindu Marriage Act, 1955)'), but NEVER distort or replace statutory titles.\n"
            "   - Do NOT translate statutory provisions in a way that alters their substantive legal meaning.\n"
            "   - If a legal concept (e.g. 'pendente lite', 'restitution of conjugal rights', 'coparcenary', 'shared household', 'stridhan') has no direct equivalent, retain the term and explain its meaning in the target language.\n"
            "   - Make it clear that your plain-language explanation is an informational summary and not an official government translation.\n"
            "4. Format your answer with these clear sections:\n"
            "   - **Direct Explanation**: Direct, objective answer to the question in the target language.\n"
            "   - **Relevant Legal Provisions**: Exact Act name and Section number from the evidence.\n"
            "   - **Plain-Language Interpretation**: What this statutory law means in plain terms.\n"
            "   - **Important Limitations & Conditions**: Procedural conditions, statutory exceptions, or jurisdictional limits.\n"
            "   - **Source Citations**: Reference sources as [1], [2], etc., matching the provided evidence.\n"
            "5. NEVER provide personalized legal advice or definitive conclusions (e.g. avoid 'you will get', 'you must'). "
            "Use objective phrasing (e.g. 'Under Section 125, an eligible spouse may petition the Family Court...').\n"
            "6. If the evidence is insufficient or partially covers the topic, state this explicitly.\n"
            f"7. Respond in the requested target language: {language.value.upper()} (en=English, hi=Hindi, kn=Kannada)."
        )

        evidence_text_blocks = []
        for i, p in enumerate(citable, start=1):
            act = p.act_name or p.document_title
            sec = f"Section {p.section_number}" if p.section_number else "Provision"
            evidence_text_blocks.append(
                f"[{i}] {act}, {sec}:\n"
                f"\"{p.legal_text.strip()}\"\n"
                f"Source: {p.source_url}"
            )

        user_prompt = (
            f"User Legal Question: \"{query}\"\n"
            f"Classified Legal Intent: {intent.value}\n"
            f"Target Language: {language.value}\n\n"
            f"--- AUTHORITATIVE RETRIEVED STATUTORY EVIDENCE ---\n"
            f"{chr(10).join(evidence_text_blocks)}\n\n"
            f"Please synthesize a grounded legal explanation in {language.value.upper()} based strictly on the above statutory evidence, adhering to all legal meaning preservation rules."
        )

        try:
            generated = self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)
            if generated:
                return generated, warnings
        except Exception as e:
            logger.warning("LLM synthesis error (%s). Falling back to template synthesis.", e)
            warnings.append(f"LLM provider error ({type(e).__name__}); using deterministic grounded template.")

        return None, warnings

    # ------------------------------------------------------------------
    # Citation & Evidence Validation
    # ------------------------------------------------------------------

    def _build_citation_registry(self, citable: List[RetrievedPassage]) -> List[Dict[str, Any]]:
        """Construct structured citation list from verified citable passages."""
        citations = []
        for i, p in enumerate(citable, start=1):
            citations.append({
                "index": i,
                "chunk_id": p.chunk_id,
                "document_id": p.document_id,
                "act_name": p.act_name or p.document_title,
                "section_number": p.section_number,
                "section_title": p.section_title,
                "page_number": p.page_start,
                "source_url": p.source_url,
                "similarity_score": round(p.score, 4),
                "verified": True,
            })
        return citations

    def _validate_citations(
        self,
        llm_reply: str,
        citable: List[RetrievedPassage],
    ) -> tuple[str, List[str]]:
        """
        Validate that citations made by the LLM refer only to verified retrieved passages.
        If hallucinated section numbers or out-of-bounds citations are detected, log warning.
        """
        warnings: List[str] = []
        verified_sections = {p.section_number.lower() for p in citable if p.section_number}
        max_idx = len(citable)

        # Check for citation brackets like [6], [7] that exceed evidence count
        bracket_citations = re.findall(r"\[(\d+)\]", llm_reply)
        for num_str in bracket_citations:
            num = int(num_str)
            if num > max_idx or num < 1:
                warnings.append(f"LLM cited out-of-bounds source index [{num}]; evidence only contains {max_idx} passages.")

        # Check for section mentions (e.g. "Section 498A" when only Section 13 was retrieved)
        mentioned_sections = re.findall(r"(?:Section|Sec\.?|§)\s*([0-9]+[A-Za-z]*)", llm_reply, re.IGNORECASE)
        for sec in mentioned_sections:
            if sec.lower() not in verified_sections:
                warnings.append(f"Note: Section {sec} was referenced in synthesis but was not part of retrieved evidence.")

        return llm_reply, warnings

    def _ensure_disclaimer(self, reply: str, language: SupportedLanguage) -> str:
        """Ensure standard statutory legal disclaimer is present at the end of the reply."""
        if "legal advice" in reply.lower() or "कानूनी सलाह" in reply or "ಕಾನೂನು ಸಲಹೆ" in reply:
            return reply

        if language == SupportedLanguage.HINDI:
            disclaimer = (
                "\n\n---\n"
                "⚠️ *अस्वीकरण: यह न्यायसेतु एआई (NyayaSetu AI) द्वारा प्रदान की गई सामान्य कानूनी जानकारी है, "
                "यह किसी अधिवक्ता की कानूनी सलाह का विकल्प नहीं है। अपनी विशिष्ट कानूनी स्थिति के लिए योग्य अधिवक्ता से परामर्श लें।*"
            )
        elif language == SupportedLanguage.KANNADA:
            disclaimer = (
                "\n\n---\n"
                "⚠️ *ಹಕ್ಕುತ್ಯಾಗ: ಇದು ನ್ಯಾಯಸೇತು ಎಐ ಒದಗಿಸಿದ ಸಾಮಾನ್ಯ ಕಾನೂನು ಮಾಹಿತಿಯಾಗಿದ್ದು, "
                "ಅರ್ಹ ವಕೀಲರ ಕಾನೂನು ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ. ನಿರ್ದಿಷ್ಟ ಪ್ರಕರಣಗಳಿಗೆ ವಕೀಲರನ್ನು ಸಂಪರ್ಕಿಸಿ.*"
            )
        else:
            disclaimer = (
                "\n\n---\n"
                "⚠️ *Disclaimer: NyayaSetu AI provides general legal information and decision support for Indian family law. "
                "It is not legal advice and does not substitute for consultation with a qualified legal practitioner.*"
            )
        return reply + disclaimer

    # ------------------------------------------------------------------
    # Deterministic Grounded Template Responses
    # ------------------------------------------------------------------

    def _no_corpus_response(self, language: SupportedLanguage) -> str:
        if language == SupportedLanguage.HINDI:
            return (
                "**[ज्ञान आधार तैयार नहीं है — NyayaSetu AI]**\n\n"
                "अभी तक कोई आधिकारिक कानूनी दस्तावेज़ अनुक्रमित (indexed) नहीं किए गए हैं। "
                "कृपया सबसे पहले `data/raw/` में भारत कोड से आधिकारिक बेयर एक्ट PDF डाउनलोड करें, "
                "फिर इन्जेशन पाइपलाइन (`python -m app.ingestion.pipeline`) और "
                "इंडेक्सर (`python -m app.rag.indexer`) चलाएं।\n\n"
                "*(यह कानूनी सलाह नहीं है। योग्य अधिवक्ता से परामर्श लें।)*"
            )
        elif language == SupportedLanguage.KANNADA:
            return (
                "**[ಜ್ಞಾನ ಆಧಾರ ಸಿದ್ಧವಾಗಿಲ್ಲ — NyayaSetu AI]**\n\n"
                "ಇನ್ನೂ ಯಾವುದೇ ಅಧಿಕೃತ ಕಾನೂನು ದಾಖಲೆಗಳನ್ನು ಇಂಡೆಕ್ಸ್ (indexed) ಮಾಡಲಾಗಿಲ್ಲ. "
                "ದಯವಿಟ್ಟು `data/raw/` ಗೆ ಭಾರತ ಕೋಡ್‌ನಿಂದ PDF ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ, "
                "ನಂತರ ingestion (`python -m app.ingestion.pipeline`) ಮತ್ತು "
                "indexer (`python -m app.rag.indexer`) ಚಲಾಯಿಸಿ.\n\n"
                "*(ಇದು ಕಾನೂನು ಸಲಹೆ ಅಲ್ಲ. ಅರ್ಹ ವಕೀಲರನ್ನು ಸಂಪರ್ಕಿಸಿ.)*"
            )
        else:
            return (
                "**[Knowledge Base Not Yet Indexed — NyayaSetu AI]**\n\n"
                "No official legal documents have been indexed yet in NyayaSetu AI. "
                "To activate retrieval, download official Bare Act PDFs from "
                "[India Code](https://www.indiacode.nic.in/) into `data/raw/`, "
                "then run the ingestion pipeline (`python -m app.ingestion.pipeline`) "
                "followed by the indexer (`python -m app.rag.indexer`).\n\n"
                "*(This is not legal advice. Consult a qualified legal practitioner.)*"
            )

    def _insufficient_evidence_response(
        self,
        query: str,
        language: SupportedLanguage,
        clarification_prompt: Optional[str],
    ) -> str:
        if language == SupportedLanguage.HINDI:
            base = (
                "**[अपर्याप्त प्रमाण — Insufficient Evidence]**\n\n"
                f"आपके प्रश्न: *\"{query}\"* के संबंध में अनुक्रमित भारतीय कानूनी डेटाबेस में कोई पर्याप्त प्रासंगिक प्रावधान नहीं मिला।\n\n"
                "**संभावित कारण:**\n"
                "• प्रश्न बहुत व्यापक है या इसमें कानूनी शब्दों का अभाव है।\n"
                "• इस विषय से संबंधित कानून अभी तक डेटाबेस में अनुक्रमित नहीं किया गया है।\n"
                "• यह विषय वर्तमान अनुक्रमित पारिवारिक कानून संग्रह से बाहर हो सकता है।\n\n"
                "*(यह कानूनी सलाह नहीं है। विशिष्ट मार्गदर्शन के लिए किसी योग्य अधिवक्ता से संपर्क करें।)*"
            )
        elif language == SupportedLanguage.KANNADA:
            base = (
                "**[ಅಪರ್ಯಾಪ್ತ ಸಾಕ್ಷ್ಯ — Insufficient Evidence]**\n\n"
                f"ನಿಮ್ಮ ಪ್ರಶ್ನೆ: *\"{query}\"* ಗೆ ಸಂಬಂಧಿಸಿದಂತೆ ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ಶಾಸನಬದ್ಧ ನಿಬಂಧನೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ.\n\n"
                "*(ಇದು ಕಾನೂನು ಸಲಹೆ ಅಲ್ಲ. ಅರ್ಹ ವಕೀಲರನ್ನು ಸಂಪರ್ಕಿಸಿ.)*"
            )
        else:
            base = (
                "**[Insufficient Evidence]**\n\n"
                f"For your query: *\"{query}\"*\n\n"
                "No indexed statutory provisions met the minimum relevance threshold. "
                "Per NyayaSetu AI safety standards, answers are not generated without supporting statutory evidence.\n\n"
                "**Possible reasons:**\n"
                "• The question is broad or phrasing requires clarification.\n"
                "• Specific statutory provisions covering this topic have not yet been indexed.\n"
                "• The issue falls outside the currently ingested Indian family law corpus.\n\n"
                "*(This is general legal information, not personalized legal advice. Consult a qualified legal practitioner.)*"
            )
        if clarification_prompt:
            base += f"\n\n---\n**Clarification Suggestion:** {clarification_prompt}"
        return base

    def _build_grounded_response(
        self,
        query: str,
        intent: QueryIntent,
        language: SupportedLanguage,
        passages: List[RetrievedPassage],
        clarification_prompt: Optional[str],
    ) -> str:
        """Compose structured 5-part grounded response from qualifying passages."""
        lines: list[str] = []

        intent_label = intent.value.replace("_", " ").title()

        # 1. Header & Direct Explanation
        if language == SupportedLanguage.HINDI:
            lines.append(f"### ⚖️ विषय: {intent_label} — भारतीय पारिवारिक कानून")
            lines.append(f"\n**प्रत्यक्ष स्पष्टीकरण (Direct Explanation):**")
            lines.append(
                f"आपके प्रश्न *\"{query}\"* के संबंध में, आधिकारिक अनुक्रमित भारतीय कानूनी संग्रह "
                f"से {len(passages)} प्रासंगिक वैधानिक प्रावधान प्राप्त हुए हैं। ये प्रावधान संबंधित अधिकारों और प्रक्रियाओं को निर्धारित करते हैं।"
            )
            lines.append(f"\n**प्रासंगिक कानूनी प्रावधान (Relevant Legal Provisions):**")
        elif language == SupportedLanguage.KANNADA:
            lines.append(f"### ⚖️ ವಿಷಯ: {intent_label} — ಭಾರತೀಯ ಕೌಟುಂಬಿಕ ಕಾನೂನು")
            lines.append(f"\n**ನೇರ ವಿವರಣೆ (Direct Explanation):**")
            lines.append(
                f"ನಿಮ್ಮ ಪ್ರಶ್ನೆ *\"{query}\"* ಗೆ ಸಂಬಂಧಿಸಿದಂತೆ, {len(passages)} ಪ್ರಮುಖ ಶಾಸನಬದ್ಧ ಪ್ರಾವಧಾನಗಳು ಲಭ್ಯವಾಗಿವೆ."
            )
            lines.append(f"\n**ಸಂಬಂಧಿತ ಕಾನೂನು ನಿಬಂಧನೆಗಳು (Relevant Legal Provisions):**")
        else:
            lines.append(f"### ⚖️ Legal Overview: {intent_label} (Indian Family Law)")
            lines.append(f"\n**1. Direct Explanation:**")
            lines.append(
                f"In response to your inquiry regarding *\"{query}\"*, the authoritative Indian legal corpus "
                f"provides specific statutory frameworks governing {intent_label.lower()}. "
                f"The following {len(passages)} statutory provision(s) were identified as directly relevant evidence."
            )
            lines.append(f"\n**2. Relevant Statutory Provisions:**")

        # 2. Statutory Provisions (Act & Section)
        for i, passage in enumerate(passages, start=1):
            act_ref = passage.act_name or passage.document_title
            sec_ref = f"Section {passage.section_number}" if passage.section_number else "Statutory Provision"
            if passage.section_title:
                sec_ref += f" ({passage.section_title})"

            lines.append(f"\n**[{i}] {act_ref} — {sec_ref}**")
            lines.append(f"> \"{passage.legal_text.strip()}\"")
            lines.append(f"*Source:* [{act_ref}]({passage.source_url}) • *Relevance Score:* {passage.score:.1%}")

        # 3. Simple-Language Interpretation
        if language == SupportedLanguage.HINDI:
            lines.append("\n**3. सरल भाषा में विधिक व्याख्या (Plain-Language Legal Interpretation):**")
            lines.append("*(यह व्याख्या सामान्य भाषा में समझ के लिए है, आधिकारिक अनुवाद नहीं है। आधिकारिक वैधानिक पाठ ऊपर उद्धृत किया गया है।)*")
            if intent == QueryIntent.DIVORCE:
                lines.append(
                    "भारतीय कानून के तहत तलाक की कार्यवाही विशिष्ट वैधानिक आधारों पर आधारित होनी चाहिए "
                    "(जैसे हिंदू विवाह अधिनियम की धारा 13B के तहत आपसी सहमति, या धारा 13 के तहत अन्य वैधानिक आधार)। "
                    "याचिका क्षेत्राधिकार वाले सक्षम पारिवारिक न्यायालय (Family Court) में प्रस्तुत की जाती है।"
                )
            elif intent == QueryIntent.MAINTENANCE:
                lines.append(
                    "भारतीय कानून (जैसे सीआरपीसी की धारा 125 या हिंदू विवाह अधिनियम की धारा 24/25) के तहत भरण-पोषण का उद्देश्य "
                    "आश्रित पक्ष को आर्थिक सुरक्षा प्रदान करना है। "
                    "पारिवारिक न्यायालय आवेदक की वास्तविक आवश्यकताओं और प्रतिवादी की आय व जीवन स्तर का मूल्यांकन करता है।"
                )
            elif intent == QueryIntent.CUSTODY:
                lines.append(
                    "बाल अभिरक्षा (Child Custody) मामलों में भारतीय न्यायालय यह मानते हैं कि "
                    "**'नाबालिग बच्चे का सर्वोत्तम कल्याण' (Paramount Welfare of the Minor Child)** सर्वोपरि सिद्धांत है।"
                )
            elif intent == QueryIntent.DOMESTIC_VIOLENCE:
                lines.append(
                    "घरेलू हिंसा से महिला संरक्षण अधिनियम, 2005 (PWDVA, 2005) पीड़ित महिलाओं को संरक्षण आदेश (Protection Orders), "
                    "निवास आदेश (Residence Orders) और मौद्रिक राहत (Monetary Relief) जैसे दीवानी उपचार प्रदान करता है।"
                )
            else:
                lines.append(
                    "प्राप्त वैधानिक प्रावधान संसद द्वारा निर्धारित कानूनी अधिकार, शर्तें और प्रक्रियाएं स्पष्ट करते हैं। "
                    "सभी कार्यवाही सक्षम पारिवारिक अथवा सिविल न्यायालय में विधिवत प्रारंभ की जानी चाहिए।"
                )
        elif language == SupportedLanguage.KANNADA:
            lines.append("\n**3. ಸರಳ ಭಾಷೆಯ ಕಾನೂನು ವ್ಯಾಖ್ಯಾನ (Plain-Language Legal Interpretation):**")
            lines.append("*(ಈ ವಿವರಣೆಯು ತಿಳುವಳಿಕೆಗಾಗಿ ಮಾತ್ರ, ಅಧಿಕೃತ ಕಾನೂನು ಅನುವಾದವಲ್ಲ. ಅಧಿಕೃತ ಶಾಸನಬದ್ಧ ಪಠ್ಯವನ್ನು ಮೇಲೆ ನೀಡಲಾಗಿದೆ.)*")
            if intent == QueryIntent.DIVORCE:
                lines.append(
                    "ಭಾರತೀಯ ಕಾನೂನಿನ ಪ್ರಕಾರ ವಿಚ್ಛೇದನ ಪ್ರಕ್ರಿಯೆಯು ನಿರ್ದಿಷ್ಟ ಶಾಸನಬದ್ಧ ಆಧಾರಗಳ ಮೇಲೆ ಇರಬೇಕು "
                    "(ಉದಾಹರಣೆಗೆ ಹಿಂದೂ ವಿವಾಹ ಕಾಯ್ದೆಯ ಸೆಕ್ಷನ್ 13B ರ ಅಡಿಯಲ್ಲಿ ಪರಸ್ಪರ ಒಪ್ಪಿಗೆ, ಅಥವಾ ಸೆಕ್ಷನ್ 13 ರ ಅಡಿಯಲ್ಲಿ ಇತರ ಆಧಾರಗಳು). "
                    "ಅರ್ಜಿಯನ್ನು ಸಕ್ಷಮ ಕೌಟುಂಬಿಕ ನ್ಯಾಯಾಲಯದಲ್ಲಿ ಸಲ್ಲಿಸಬೇಕು."
                )
            elif intent == QueryIntent.MAINTENANCE:
                lines.append(
                    "ಭಾರತೀಯ ಕಾನೂನಿನಡಿ ಜೀವನಾಂಶವು (ಉದಾಹರಣೆಗೆ ಸೆಕ್ಷನ್ 125 ಸಿಆರ್‌ಪಿಸಿ ಅಥವಾ ಸೆಕ್ಷನ್ 24/25 ಎಚ್‌ಎಂಎ) ಜೀವನ ನಿರ್ವಹಣೆಗೆ ಆರ್ಥಿಕ ಬೆಂಬಲ ನೀಡಲು ಉದ್ದೇಶಿಸಲಾಗಿದೆ. "
                    "ಅರ್ಜಿದಾರರ ಸಮಂಜಸ ಅಗತ್ಯಗಳು ಮತ್ತು ಎದುರುದಾರರ ಆರ್ಥಿಕ ಸಾಮರ್ಥ್ಯವನ್ನು ನ್ಯಾಯಾಲಯ ಪರಿಶೀಲಿಸುತ್ತದೆ."
                )
            elif intent == QueryIntent.CUSTODY:
                lines.append(
                    "ಮಕ್ಕಳ ಪಾಲನೆ ವಿಷಯಗಳಲ್ಲಿ (ಗಾರ್ಡಿಯನ್ಸ್ ಆ್ಯಂಡ್ ವಾರ್ಡ್ಸ್ ಆ್ಯಕ್ಟ್), "
                    "**'ಅಪ್ರಾಪ್ತ ಮಗುವಿನ ಕ್ಷೇಮಾಭಿವೃದ್ಧಿ' (Paramount Welfare of the Minor Child)** ಅತ್ಯಂತ ಪ್ರಮುಖ ತತ್ವವೆಂದು ನ್ಯಾಯಾಲಯಗಳು ಪರಿಗಣಿಸುತ್ತವೆ."
                )
            elif intent == QueryIntent.DOMESTIC_VIOLENCE:
                lines.append(
                    "ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ ತಡೆ ಕಾಯ್ದೆ, 2005 (PWDVA, 2005) ರಕ್ಷಣಾ ಆದೇಶಗಳು (Protection Orders), "
                    "ವಾಸಸ್ಥಳ ಆದೇಶಗಳು (Residence Orders) ಮತ್ತು ಆರ್ಥಿಕ ಪರಿಹಾರಗಳನ್ನು ಒದಗಿಸುತ್ತದೆ."
                )
            else:
                lines.append(
                    "ಲಭ್ಯವಾದ ಶಾಸನಬದ್ಧ ನಿಬಂಧನೆಗಳು ಭಾರತೀಯ ಸಂಸತ್ತು ರೂಪಿಸಿದ ಕಾನೂನು ಹಕ್ಕುಗಳು ಮತ್ತು ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ತಿಳಿಸುತ್ತವೆ. "
                    "ಎಲ್ಲಾ ಪ್ರಕ್ರಿಯೆಗಳನ್ನು ಸಂಬಂಧಪಟ್ಟ ಕೌಟುಂಬಿಕ ನ್ಯಾಯಾಲಯದಲ್ಲಿ ಆರಂಭಿಸಬೇಕು."
                )
        else:
            lines.append("\n**3. Plain-Language Interpretation:**")
            lines.append("*(This interpretation is a plain-language explanation and decision-support summary, not an official statutory translation.)*")
            if intent == QueryIntent.DIVORCE:
                lines.append(
                    "Under Indian statutory law, divorce proceedings must be grounded in specific statutory criteria "
                    "(such as mutual consent under Section 13B of the Hindu Marriage Act, or contested grounds under Section 13). "
                    "Petitions are adjudicated by the Family Court having territorial jurisdiction."
                )
            elif intent == QueryIntent.MAINTENANCE:
                lines.append(
                    "Maintenance under Indian law (such as Section 125 CrPC or Section 24/25 HMA) is designed to prevent vagrancy and destitution. "
                    "The Family Court evaluates the financial capacity of the respondent, the standard of living, and the reasonable needs of the applicant."
                )
            elif intent == QueryIntent.CUSTODY:
                lines.append(
                    "In child custody matters (under the Guardians and Wards Act, 1890), Indian courts hold that the "
                    "**paramount welfare of the minor child** supersedes all other considerations, including parental claims."
                )
            elif intent == QueryIntent.DOMESTIC_VIOLENCE:
                lines.append(
                    "The Protection of Women from Domestic Violence Act, 2005 provides civil remedies including Protection Orders, "
                    "Residence Orders, and Monetary Relief through an application to the Judicial Magistrate."
                )
            else:
                lines.append(
                    "The retrieved provisions establish the statutory rights, conditions, and procedures established by the Parliament of India. "
                    "All proceedings must be initiated in the competent court of civil or family jurisdiction."
                )

        # 4. Important Limitations & Conditions
        if language == SupportedLanguage.HINDI:
            lines.append("\n**4. महत्वपूर्ण सीमाएं एवं प्रक्रियात्मक शर्तें (Important Limitations & Procedural Conditions):**")
            lines.append("• **अधिकार क्षेत्र (Jurisdiction):** पारिवारिक अदालत का अधिकार क्षेत्र विवाह के स्थान, अंतिम साझा निवास, या प्रतिवादी के निवास पर निर्भर करता है।")
            lines.append("• **न्यायिक विवेकाधिकार (Judicial Discretion):** वैधानिक प्रावधान अधिकार स्थापित करते हैं, परंतु वास्तविक राहत साक्ष्यों के आधार पर अदालत के विवेकाधिकार पर निर्भर है।")
            lines.append("• **राज्य संशोधन (State Amendments):** कुछ वैधानिक प्रावधानों में राज्य-विशिष्ट प्रक्रियात्मक संशोधन लागू हो सकते हैं।")
            lines.append("\n**5. आधिकारिक वैधानिक उद्धरण (Official Statutory Citations):**")
        elif language == SupportedLanguage.KANNADA:
            lines.append("\n**4. ಪ್ರಮುಖ ಮಿತಿಗಳು ಮತ್ತು ಷರತ್ತುಗಳು (Important Limitations & Procedural Conditions):**")
            lines.append("• **ನ್ಯಾಯವ್ಯಾಪ್ತಿ (Jurisdiction):** ಕೌಟುಂಬಿಕ ನ್ಯಾಯಾಲಯದ ಅಧಿಕಾರ ವ್ಯಾಪ್ತಿಯು ವಿವಾಹ ನಡೆದ ಸ್ಥಳ, ಜಂಟಿ ವಾಸಸ್ಥಳ ಅಥವಾ ಪ್ರತಿವಾದಿಯ ವಾಸಸ್ಥಳವನ್ನು ಅವಲಂಬಿಸಿರುತ್ತದೆ.")
            lines.append("• **ನ್ಯಾಯಾಂಗ ವಿವೇಚನೆ (Judicial Discretion):** ಶಾಸನಗಳು ಹಕ್ಕುಗಳನ್ನು ವಿವರಿಸುತ್ತವೆ, ಆದರೆ ಅಂತಿಮ ಪರಿಹಾರವು ಸಾಕ್ಷ್ಯಾಧಾರಗಳ ಮೇಲಿನ ನ್ಯಾಯಾಲಯದ ವಿವೇಚನೆಗೆ ಒಳಪಟ್ಟಿರುತ್ತದೆ.")
            lines.append("• **ರಾಜ್ಯ ತಿದ್ದುಪಡಿಗಳು (State Amendments):** ಕೆಲವು ನಿಬಂಧನೆಗಳಿಗೆ ರಾಜ್ಯ-ನಿರ್ದಿಷ್ಟ ಪ್ರಕ್ರಿಯಾತ್ಮಕ ತಿದ್ದುಪಡಿಗಳು ಅನ್ವಯಿಸಬಹುದು.")
            lines.append("\n**5. ಅಧಿಕೃತ ಶಾಸನಬದ್ಧ ಉಲ್ಲೇಖಗಳು (Official Statutory Citations):**")
        else:
            lines.append("\n**4. Important Limitations & Procedural Conditions:**")
            lines.append("• **Jurisdiction:** Family court jurisdiction depends on the marriage place, shared residence, or respondent's residence.")
            lines.append("• **Discretionary Evaluation:** Statutory provisions outline legal rights, but actual relief is subject to judicial discretion based on submitted evidence.")
            lines.append("• **State Amendments:** Certain statutory provisions may carry state-specific procedural amendments.")
            lines.append("\n**5. Official Statutory Citations:**")

        for i, passage in enumerate(passages, start=1):
            act_ref = passage.act_name or passage.document_title
            lines.append(f"[{i}] [{act_ref}]({passage.source_url}) (India Code Official Repository)")

        if clarification_prompt:
            lines.append(f"\n---\n**Clarification Note:** {clarification_prompt}")

        # Safety Footer
        if language == SupportedLanguage.HINDI:
            lines.append(
                "\n---\n"
                "⚠️ *अस्वीकरण (Disclaimer): न्यायसेतु एआई (NyayaSetu AI) भारतीय पारिवारिक कानून के लिए सामान्य कानूनी जानकारी और निर्णय-सहायता प्रदान करता है। "
                "यह किसी योग्य अधिवक्ता की व्यक्तिगत कानूनी सलाह का विकल्प नहीं है।*"
            )
        elif language == SupportedLanguage.KANNADA:
            lines.append(
                "\n---\n"
                "⚠️ *ಹಕ್ಕುತ್ಯಾಗ (Disclaimer): ನ್ಯಾಯಸೇತು ಎಐ (NyayaSetu AI) ಭಾರತೀಯ ಕೌಟುಂಬಿಕ ಕಾನೂನಿನ ಬಗ್ಗೆ ಸಾಮಾನ್ಯ ಕಾನೂನು ಮಾಹಿತಿ ನೀಡುತ್ತದೆ. "
                "ಇದು ಅರ್ಹ ವಕೀಲರ ವೈಯಕ್ತಿಕ ಕಾನೂನು ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ.*"
            )
        else:
            lines.append(
                "\n---\n"
                "⚠️ *Disclaimer: NyayaSetu AI provides general legal information and decision support for Indian family law. "
                "It does not provide definitive personalized legal advice and does not substitute for consultation with a qualified legal practitioner.*"
            )

        return "\n".join(lines)


# Singleton
synthesizer = LegalSynthesizer()
