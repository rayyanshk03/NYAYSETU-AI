"""Query planner: intent classification and retrieval plan generation.

Classifies the user's legal query into a structured intent and builds
a retrieval plan with targeted sub-queries and optional document filters.
No external model calls are made in this step — classification uses
keyword heuristics that are transparent, fast, and inspectable.

Improvements in Step 5:
- Added GUARDIANSHIP intent distinct from CUSTODY.
- Added user_intent_type field (explain/procedure/eligibility/rights/provisions).
- Added topic field for display in the agent trace.
- Smarter clarification: fires for multi-intent ambiguity and very-short queries.
- Sub-query text deduplication: never sends the same query text twice to the retriever.
- Multi-sub-query only when genuinely useful (second Act document exists).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.chat import SupportedLanguage


class QueryIntent(str, Enum):
    """Primary intent categories for Indian family law queries."""

    DIVORCE = "divorce"
    MAINTENANCE = "maintenance"
    CUSTODY = "custody"
    GUARDIANSHIP = "guardianship"
    DOMESTIC_VIOLENCE = "domestic_violence"
    ADOPTION = "adoption"
    MARRIAGE = "marriage"
    INHERITANCE = "inheritance"
    SUCCESSION = "succession"
    GENERAL = "general"


class UserIntentType(str, Enum):
    """Fine-grained user intent within a legal topic."""

    EXPLAIN = "explain"           # What is / explain
    PROCEDURE = "procedure"       # How to / steps
    ELIGIBILITY = "eligibility"   # Am I eligible / who can
    RIGHTS = "rights"             # What are my rights
    PROVISIONS = "provisions"     # What does the law say / which sections
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Keyword maps for heuristic intent classification
# Covers English, transliterated Hindi and transliterated Kannada keywords.
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: dict[QueryIntent, list[str]] = {
    QueryIntent.DIVORCE: [
        # English
        "divorce", "dissolution", "separated", "separation", "grounds for divorce",
        "grounds of divorce", "divorce grounds", "divorce petition", "contested divorce",
        "mutual divorce", "divorce law", "mutual consent", "judicial separation",
        "nullity", "void marriage", "annulment", "decree of divorce",
        # Hindi transliterated
        "talaak", "talak", "vivah viched", "tiraskaar",
        # Hindi Devanagari
        "तलाक", "विवाह विच्छेद", "तलाक के आधार", "तलाक की प्रक्रिया",
        "पारस्परिक सहमति", "न्यायिक पृथक्करण", "शून्य विवाह", "तलाक याचिका",
        # Kannada transliterated
        "vichhedane", "vivaha vichhedane",
        # Kannada script
        "ವಿಚ್ಛೇದನ", "ವಿವಾಹ ವಿಚ್ಛೇದನ", "ವಿಚ್ಛೇದನದ ಆಧಾರ", "ವಿಚ್ಛೇದನಕ್ಕೆ",
        "ಪರಸ್ಪರ ಒಪ್ಪಿಗೆ", "ನ್ಯಾಯಾಂಗ ಪ್ರತ್ಯೇಕತೆ",
    ],
    QueryIntent.MAINTENANCE: [
        # English
        "maintenance", "alimony", "interim maintenance", "section 125",
        "bnss 144", "crpc 125", "support", "sustenance", "monthly allowance",
        "financial support", "nafqa", "permanent alimony", "pendente lite",
        "maintenance law", "maintenance claim", "claim maintenance", "maintenance for wife",
        # Hindi transliterated
        "bharan poshan", "guzara", "nafkha",
        # Hindi Devanagari
        "भरण-पोषण", "भरण पोषण", "गुजारा", "गुजारा भत्ता", "अंतरिम भरण-पोषण",
        "धारा 125", "बीएनएसएस 144", "मासिक भत्ता", "पत्नी का भरण पोषण",
        # Kannada transliterated
        "janivari", "poshan",
        # Kannada script
        "ಜೀವನಾಂಶ", "ಮಧ್ಯಂತರ ಜೀವನಾಂಶ", "ಪೋಷಣೆ", "ಸೆಕ್ಷನ್ 125", "ಮಾಸಿಕ ಭತ್ಯೆ",
        "ಹೆಂಡತಿಗೆ ಜೀವನಾಂಶ",
    ],
    QueryIntent.CUSTODY: [
        # English
        "custody", "parental rights", "visitation", "access rights",
        "welfare of child", "minor child", "custody order", "joint custody",
        "sole custody", "child custody",
        # Hindi transliterated
        "bacche ki custody", "bachche ka adhikar",
        # Hindi Devanagari
        "बाल अभिरक्षा", "कस्टडी", "बच्चे की कस्टडी", "बच्चे की देखरेख",
        "मुलाकात का अधिकार", "बच्चे का कल्याण",
        # Kannada transliterated
        "magu samrakshane", "magu custody",
        # Kannada script
        "ಮಕ್ಕಳ ಪಾಲನೆ", "ಕಸ್ಟಡಿ", "ಮಗುವಿನ ಪಾಲನೆ", "ಮಗುವಿನ ಕ್ಷೇಮಾಭಿವೃದ್ಧಿ", "ಭೇಟಿ ಹಕ್ಕು",
    ],
    QueryIntent.GUARDIANSHIP: [
        # English
        "guardianship", "guardian", "guardian and wards", "guardian ward act",
        "natural guardian", "legal guardian", "testamentary guardian",
        "child care", "welfare of minor",
        # Hindi transliterated
        "samrakshan", "abhivavak",
        # Hindi Devanagari
        "संरक्षकता", "संरक्षक", "अभिभावक", "प्राकृतिक संरक्षक", "नाबालिग का कल्याण",
        # Kannada transliterated
        "palakathva",
        # Kannada script
        "ಪಾಲಕತ್ವ", "ಪೋಷಕತ್ವ", "ನೈಸರ್ಗಿಕ ಪಾಲಕ", "ಅಪ್ರಾಪ್ತ ವಯಸ್ಕ",
    ],
    QueryIntent.DOMESTIC_VIOLENCE: [
        # English
        "domestic violence", "dv act", "protection order", "residence order",
        "abuse", "cruelty", "harassment", "shelter home", "shared household",
        "pwdva", "women domestic", "violence at home",
        # Hindi transliterated
        "gharelu hinsa", "ghar mein hinsa", "hinsa",
        # Hindi Devanagari
        "घरेलू हिंसा", "संरक्षण आदेश", "निवास आदेश", "साझा गृहस्थी", "क्रूरता",
        "प्रताड़ना", "महिला संरक्षण",
        # Kannada transliterated
        "gruha himse", "kutumba himse",
        # Kannada script
        "ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ", "ಕೌಟುಂಬಿಕ ಹಿಂಸಾಚಾರ", "ರಕ್ಷಣಾ ಆದೇಶ", "ವಾಸದ ಆದೇಶ",
    ],
    QueryIntent.ADOPTION: [
        # English
        "adopt", "adoption", "hama", "orphan", "foster",
        "cara", "central adoption", "child adoption",
        # Hindi transliterated
        "godh lena", "godh", "grahit putra",
        # Hindi Devanagari
        "दत्तक ग्रहण", "गोद लेना", "गोद", "दत्तक", "कारा",
        # Kannada transliterated
        "daththu", "daththu putra",
        # Kannada script
        "ದತ್ತು", "ದತ್ತು ಸ್ವೀಕಾರ", "ದತ್ತು ಪುತ್ರ",
    ],
    QueryIntent.MARRIAGE: [
        # English
        "marriage", "wedding", "nikah", "registration of marriage", "validity of marriage",
        "valid marriage", "age of marriage", "conditions for marriage", "conditions for a valid marriage",
        "ceremonies", "marriage certificate", "solemnization", "court marriage",
        # Hindi transliterated
        "vivah", "shadi", "shaadi", "vivah panjikaran",
        # Hindi Devanagari
        "विवाह", "शादी", "विवाह पंजीकरण", "वैध विवाह", "विवाह की शर्तें", "विवाह प्रमाण पत्र", "कोर्ट मैरिज",
        # Kannada transliterated
        "vivaha", "maduve", "nondani",
        # Kannada script
        "ವಿವಾಹ", "ಮದುವೆ", "ವಿವಾಹ ನೋಂದಣಿ", "ಮಾನ್ಯ ವಿವಾಹ", "ವಿವಾಹದ ಷರತ್ತುಗಳು",
    ],
    QueryIntent.INHERITANCE: [
        # English
        "inheritance", "hsa", "will", "testament",
        "ancestral", "coparcenary", "stridhan", "dowry property", "intestate",
        "hindu succession act",
        # Hindi transliterated
        "virasat", "vasiyat",
        # Hindi Devanagari
        "विरासत", "वसीयत", "पैतृक संपत्ति", "स्त्रीधन",
        # Kannada transliterated
        "uttaradhikar", "aasthi",
        # Kannada script
        "ಆಸ್ತಿ ಹಕ್ಕು", "ವಾರಸುದಾರಿಕೆ", "ಸ್ತ್ರೀಧನ", "ಉಯಿಲು",
    ],
    QueryIntent.SUCCESSION: [
        # English
        "succession", "property rights", "class of heirs", "legal heir", "heir",
        "estate", "testamentary succession", "succession certificate",
        # Hindi transliterated
        "sampatti", "varisan",
        # Hindi Devanagari
        "उत्तराधिकार", "उत्तराधिकारी", "संपत्ति अधिकार", "उत्तराधिकार प्रमाण पत्र",
        # Kannada transliterated
        "uttaradhikari", "aasthi hakami",
        # Kannada script
        "ಉತ್ತರಾಧಿಕಾರ", "ಉತ್ತರಾಧಿಕಾರಿ", "ಆಸ್ತಿ ಹಂಚಿಕೆ",
    ],
}

# ---------------------------------------------------------------------------
# User intent type keywords (fine-grained intent within topic)
# ---------------------------------------------------------------------------
_USER_INTENT_KEYWORDS: dict[UserIntentType, list[str]] = {
    UserIntentType.EXPLAIN: [
        "what is", "explain", "meaning of", "define", "kya hai", "arth", "enu",
        "क्या है", "अर्थ", "परिभाषा", "समझाएं", "ಏನು", "ಅರ್ಥ", "ವಿವರಿಸಿ",
    ],
    UserIntentType.PROCEDURE: [
        "how to", "steps", "process", "procedure", "file", "apply", "kaise",
        "tarike", "steps for", "filing",
        "प्रक्रिया", "कैसे", "तरीके", "दायर", "आवेदन",
        "ಹೇಗೆ", "ವಿಧಾನ", "ಪ್ರಕ್ರಿಯೆ", "ಅರ್ಜಿ",
    ],
    UserIntentType.ELIGIBILITY: [
        "eligible", "eligibility", "qualify", "who can", "can i", "kya main",
        "patrata", "yogyata",
        "पात्रता", "योग्यता", "कौन", "क्या मैं",
        "ಅರ್ಹತೆ", "ಯಾರು", "ಸಾಧ್ಯವೇ",
    ],
    UserIntentType.RIGHTS: [
        "rights", "entitlement", "entitled", "what are my rights", "adhikar",
        "hakku",
        "अधिकार", "हक", "ಹಕ್ಕುಗಳು", "ಹಕ್ಕು",
    ],
    UserIntentType.PROVISIONS: [
        "provisions", "section", "act", "law says", "statute", "which law",
        "under which section", "dhara", "kanoon",
        "धारा", "कानून", "प्रावधान", "अधिनियम",
        "ಕಾಯ್ದೆ", "ಕಾನೂನು", "ನಿಬಂಧನೆ", "ಸೆಕ್ಷನ್",
    ],
}

# Map intent → document IDs that are most relevant (used for focused retrieval)
_INTENT_DOCUMENT_HINTS: dict[QueryIntent, list[str]] = {
    QueryIntent.DIVORCE: ["hma_1955", "sma_1954"],
    QueryIntent.MAINTENANCE: ["hama_1956", "pwdva_2005"],
    QueryIntent.CUSTODY: ["hama_1956"],
    QueryIntent.GUARDIANSHIP: ["hama_1956"],
    QueryIntent.DOMESTIC_VIOLENCE: ["pwdva_2005", "hama_1956"],
    QueryIntent.ADOPTION: ["hama_1956"],
    QueryIntent.MARRIAGE: ["hma_1955", "sma_1954"],
    QueryIntent.INHERITANCE: ["hma_1955"],
    QueryIntent.SUCCESSION: ["hma_1955"],
    QueryIntent.GENERAL: [],
}


class SubQuery(BaseModel):
    """A targeted sub-query to feed into the retriever."""

    text: str = Field(..., description="Reformulated retrieval sub-query")
    document_hint: Optional[str] = Field(
        default=None,
        description="Optional document_id to prioritise in retrieval",
    )
    top_k: int = Field(default=3, description="How many passages to retrieve for this sub-query")


class QueryPlan(BaseModel):
    """Structured execution plan produced by the planner for a user query."""

    original_query: str
    language: SupportedLanguage
    intent: QueryIntent
    topic: str = Field(
        default="general",
        description="Human-readable legal topic label (e.g. 'Divorce', 'Maintenance')",
    )
    user_intent_type: UserIntentType = Field(
        default=UserIntentType.UNKNOWN,
        description="Fine-grained user intent: explain / procedure / eligibility / rights / provisions",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heuristic confidence in the classified intent",
    )
    sub_queries: List[SubQuery] = Field(default_factory=list)
    needs_clarification: bool = Field(default=False)
    clarification_prompt: Optional[str] = Field(default=None)
    planner_notes: List[str] = Field(default_factory=list)


class QueryPlanner:
    """
    Heuristic query planner for NyayaSetu AI.

    Classifies legal topic intent using multilingual keyword matching,
    identifies the user's fine-grained request type (explain/procedure/
    eligibility/rights/provisions), builds deduplicated retrieval sub-queries,
    and determines whether a clarification question is genuinely necessary.

    Rules:
    - Sub-query texts are deduplicated: the same text string is never sent twice.
    - A secondary sub-query (different document filter) is only added when a
      second Act document hint exists AND the query requires cross-Act context.
    - A broad fallback sub-query (no document filter) is only added for GENERAL
      intent or when confidence is low.
    - Clarification is requested only when:
        (a) intent is GENERAL with very low confidence, or
        (b) the query is fewer than 4 words and no specific intent was found.
    """

    def plan(self, query: str, language: SupportedLanguage) -> QueryPlan:
        """
        Produce a QueryPlan for a raw user legal query.

        Args:
            query: Raw user query string.
            language: Declared query language.

        Returns:
            QueryPlan with intent, topic, user_intent_type, sub-queries, and planner metadata.
        """
        query_lower = query.lower()
        notes: list[str] = []

        # --- Step 1: Classify legal topic intent ---
        intent, confidence = self._classify_intent(query_lower)
        topic = intent.value.replace("_", " ").title()
        notes.append(f"Classified intent: {intent.value} (confidence={confidence:.2f})")

        # --- Step 2: Classify fine-grained user intent type ---
        user_intent_type = self._classify_user_intent_type(query_lower)
        notes.append(f"User intent type: {user_intent_type.value}")

        # --- Step 3: Build deduplicated sub-queries ---
        sub_queries = self._build_sub_queries(query, intent, confidence, language)
        notes.append(
            f"Generated {len(sub_queries)} sub-quer{'y' if len(sub_queries)==1 else 'ies'} "
            f"(query-text deduplication applied)"
        )

        # --- Step 4: Determine clarification need ---
        needs_clarification, clarification_prompt = self._check_clarification(
            query, intent, confidence, language
        )
        if needs_clarification:
            notes.append("Clarification prompt attached: query lacks specific legal context")

        return QueryPlan(
            original_query=query,
            language=language,
            intent=intent,
            topic=topic,
            user_intent_type=user_intent_type,
            confidence=confidence,
            sub_queries=sub_queries,
            needs_clarification=needs_clarification,
            clarification_prompt=clarification_prompt,
            planner_notes=notes,
        )

    # ------------------------------------------------------------------
    # Intent Classification
    # ------------------------------------------------------------------

    def _classify_intent(self, query_lower: str) -> tuple[QueryIntent, float]:
        """Score each intent using keyword hit counts and return the winner."""
        scores: dict[QueryIntent, int] = {intent: 0 for intent in QueryIntent}

        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                # Phrase match scores higher than single-word match
                if " " in kw:
                    if kw in query_lower:
                        scores[intent] += 2
                elif not kw.isascii():
                    # For Indic scripts (Devanagari, Kannada), agglutinative suffixes attach directly
                    if kw in query_lower:
                        scores[intent] += 2 if len(kw) >= 4 else 1
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                        scores[intent] += 1

        best_intent = max(scores, key=lambda i: scores[i])
        best_score = scores[best_intent]

        if best_score == 0:
            return QueryIntent.GENERAL, 0.15

        # Normalise to a rough confidence: 1 hit ~ 0.42, 2 hits ~ 0.64, 3+ hits ~ 0.86+
        confidence = min(1.0, 0.20 + (best_score * 0.22))
        return best_intent, round(confidence, 2)

    def _classify_user_intent_type(self, query_lower: str) -> UserIntentType:
        """Identify fine-grained intent type (explain/procedure/eligibility/rights/provisions)."""
        for intent_type, keywords in _USER_INTENT_KEYWORDS.items():
            for kw in keywords:
                if " " in kw or not kw.isascii():
                    if kw in query_lower:
                        return intent_type
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                        return intent_type
        return UserIntentType.UNKNOWN

    # ------------------------------------------------------------------
    # Sub-Query Construction
    # ------------------------------------------------------------------

    def _build_sub_queries(
        self,
        original_query: str,
        intent: QueryIntent,
        confidence: float,
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
    ) -> list[SubQuery]:
        """
        Assemble focused, deduplicated sub-queries for the retriever.

        Strategy:
        1. Primary sub-query: original query text with the primary document hint.
        2. Secondary sub-query: original query text with a DIFFERENT document hint.
           Only emitted when a second hint exists (cross-Act scenarios).
           The query text matches primary, so instead use a topic-focused restatement.
        3. Broad fallback: only for GENERAL intent or very low confidence.

        Deduplication: seen_texts tracks query texts; a sub-query is skipped if
        its (text, document_hint) combination was already added.
        """
        sub_queries: list[SubQuery] = []
        seen_texts: set[str] = set()
        doc_hints = _INTENT_DOCUMENT_HINTS.get(intent, [])

        # 1. Primary sub-query (retains the exact original query)
        primary_doc = doc_hints[0] if doc_hints else None
        sub_queries.append(
            SubQuery(text=original_query, document_hint=primary_doc, top_k=4)
        )
        seen_texts.add(original_query.lower())

        # 2. Secondary sub-query against second Act document (cross-Act queries)
        if len(doc_hints) > 1:
            if language == SupportedLanguage.HINDI:
                secondary_text = f"{original_query} कानूनी प्रावधान"
            elif language == SupportedLanguage.KANNADA:
                secondary_text = f"{original_query} ಶಾಸನಬದ್ಧ ನಿಬಂಧನೆಗಳು"
            else:
                secondary_text = f"{original_query} statutory provisions"

            if secondary_text.lower() not in seen_texts:
                sub_queries.append(
                    SubQuery(text=secondary_text, document_hint=doc_hints[1], top_k=2)
                )
                seen_texts.add(secondary_text.lower())

        # 3. Broad fallback — only when genuinely needed
        if intent == QueryIntent.GENERAL or confidence < 0.3:
            if original_query.lower() not in seen_texts:
                sub_queries.append(
                    SubQuery(text=original_query, document_hint=None, top_k=3)
                )
                seen_texts.add(original_query.lower())
            else:
                if language == SupportedLanguage.HINDI:
                    fallback_text = f"{original_query} विधिक सिद्धांत"
                elif language == SupportedLanguage.KANNADA:
                    fallback_text = f"{original_query} ಕಾನೂನು ತತ್ವಗಳು"
                else:
                    fallback_text = f"{original_query} statutory principles"

                if fallback_text.lower() not in seen_texts:
                    sub_queries.append(
                        SubQuery(text=fallback_text, document_hint=None, top_k=3)
                    )
                    seen_texts.add(fallback_text.lower())

        # Strict deduplication by lowercased stripped query text
        unique_sub_queries: list[SubQuery] = []
        dedup_texts: set[str] = set()
        for sq in sub_queries:
            key = sq.text.strip().lower()
            if key not in dedup_texts:
                dedup_texts.add(key)
                unique_sub_queries.append(sq)

        return unique_sub_queries

    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------

    def _check_clarification(
        self,
        query: str,
        intent: QueryIntent,
        confidence: float,
        language: SupportedLanguage,
    ) -> tuple[bool, Optional[str]]:
        """
        Return (needs_clarification, clarification_prompt).

        Clarification is triggered when:
        - Intent is GENERAL with very low confidence (< 0.25), or
        - The query is fewer than 4 meaningful words and no specific intent was found.
        """
        word_count = len(query.strip().split())
        is_ambiguous = (
            intent == QueryIntent.GENERAL and confidence < 0.25
        ) or (
            word_count < 4 and intent == QueryIntent.GENERAL
        )

        if is_ambiguous:
            return True, self._build_clarification_prompt(language)
        return False, None

    def _build_clarification_prompt(self, language: SupportedLanguage) -> str:
        """Return a language-appropriate clarification question."""
        if language == SupportedLanguage.HINDI:
            return (
                "आपका प्रश्न थोड़ा सामान्य है। कृपया बताएं: क्या आप तलाक, भरण-पोषण, "
                "बाल अभिरक्षा, घरेलू हिंसा, दत्तक ग्रहण, या विवाह से संबंधित "
                "किसी विशेष कानूनी विषय के बारे में जानकारी चाहते हैं?"
            )
        elif language == SupportedLanguage.KANNADA:
            return (
                "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಸ್ವಲ್ಪ ಸಾಮಾನ್ಯವಾಗಿದೆ. ದಯವಿಟ್ಟು ನಿರ್ದಿಷ್ಟಪಡಿಸಿ: \n"
                "ವಿಚ್ಛೇದನ, ಜೀವನಾಂಶ, ಮಗುವಿನ ಸಂರಕ್ಷಣೆ, ಗೃಹ ಹಿಂಸೆ, ದತ್ತು ಸ್ವೀಕಾರ, "
                "ಅಥವಾ ವಿವಾಹ ನೋಂದಣಿ — ಯಾವ ವಿಷಯದ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕು?"
            )
        else:
            return (
                "Your query is broad. To provide the most relevant statutory information, "
                "could you clarify which area of Indian family law you need help with?\n"
                "Options: **Divorce**, **Maintenance/Alimony**, **Child Custody**, "
                "**Guardianship**, **Domestic Violence**, **Adoption**, "
                "**Marriage Registration**, or **Inheritance/Succession**."
            )


# Convenience singleton
planner = QueryPlanner()
