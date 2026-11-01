"""Language detection and resolution service for NyayaSetu AI.

Implements robust script-based and linguistic token detection for:
- English (en)
- Hindi (hi) — Devanagari script and Latin transliteration
- Kannada (kn) — Kannada script and Latin transliteration

Priority Rule:
An explicit user-selected language ALWAYS overrides automatic detection.
Automatic detection runs only when the user has not explicitly chosen a language
(or selected 'auto'). If detection is uncertain, the system flags the uncertainty
so the orchestrator can request explicit language selection from the user.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple, Optional
from pydantic import BaseModel, Field

from app.schemas.chat import SupportedLanguage

logger = logging.getLogger(__name__)

# Unicode character ranges
# Devanagari: U+0900 to U+097F
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Kannada: U+0C80 to U+0CFF
_KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
# Latin characters
_LATIN_RE = re.compile(r"[a-zA-Z]")


# Transliterated Hindi legal & common keywords (Latin script)
_HINDI_TRANSLIT_KEYWORDS = {
    "talaak", "talak", "vivah", "shadi", "shaadi", "bharan", "poshan",
    "guzara", "nafkha", "adhikar", "kanoon", "kya", "kaise", "karein",
    "hinsa", "bachche", "bacche", "patni", "pati", "godh", "virasat",
    "vasiyat", "dhara", "samrakshan", "abhivavak", "sampatti", "varisan",
    "gharelu", "nirdesh", "niyam", "yogyata", "patrata", "arth", "kripya",
    "bataye", "mujhe", "chahiye", "adalat", "vakil", "sahayata", "tarike",
}

# Transliterated Kannada legal & common keywords (Latin script)
_KANNADA_TRANSLIT_KEYWORDS = {
    "vichhedane", "maduve", "samrakshane", "palakathva", "magu", "hendathi",
    "ganda", "enu", "yava", "kanonigaive", "daththu", "aasthi", "hakku",
    "himse", "nondani", "kaayde", "kanunu", "kannada", "bagge", "thilisi",
    "hege", "padiyuvudu", "poshan", "janivari", "uttaradhikari", "arji",
    "nyayalaya", "vidhana", "salaha", "sahaya",
}

# English common & legal query keywords
_ENGLISH_KEYWORDS = {
    "what", "how", "when", "where", "why", "who", "which", "can", "is", "are",
    "the", "and", "or", "under", "for", "with", "from", "in", "to", "of",
    "divorce", "maintenance", "custody", "marriage", "alimony", "husband",
    "wife", "child", "children", "minor", "domestic", "violence", "act",
    "section", "grounds", "procedure", "rights", "petition", "court",
    "protection", "order", "hindu", "special", "guardian", "adoption",
    "inheritance", "succession", "property", "valid", "interim", "remedy",
    "legal", "law", "lawyer", "process", "eligibility", "conditions",
}


class LanguageDetectionResult(BaseModel):
    """Result of language detection on raw query text."""

    detected_language: Optional[SupportedLanguage] = Field(
        default=None,
        description="Detected language if confident, None if uncertain",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of detection (0.0 to 1.0)",
    )
    is_uncertain: bool = Field(
        default=False,
        description="True if language cannot be identified with acceptable confidence",
    )
    method: str = Field(
        default="uncertain",
        description="Detection method used: script_devanagari, script_kannada, latin_english, latin_transliterated_hindi, latin_transliterated_kannada, uncertain",
    )


class LanguageResolution(NamedTuple):
    """Final resolved language with provenance and uncertainty flags."""

    language: SupportedLanguage
    is_explicit: bool
    is_uncertain: bool
    detection: LanguageDetectionResult


class LanguageDetector:
    """Robust multilingual language detector for English, Hindi, and Kannada."""

    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect whether text is English, Hindi, or Kannada.

        Rules:
        1. Native Devanagari script (\u0900-\u097F) -> Hindi (confidence >= 0.95)
        2. Native Kannada script (\u0C80-\u0CFF) -> Kannada (confidence >= 0.95)
        3. Latin script -> Analyzed for English vs. transliterated Hindi/Kannada tokens
        4. Insufficient, ambiguous, or non-target script text -> is_uncertain = True
        """
        clean = (text or "").strip()
        if not clean:
            return LanguageDetectionResult(
                detected_language=None,
                confidence=0.0,
                is_uncertain=True,
                method="empty_input",
            )

        # Count script character occurrences
        devanagari_count = len(_DEVANAGARI_RE.findall(clean))
        kannada_count = len(_KANNADA_RE.findall(clean))
        latin_count = len(_LATIN_RE.findall(clean))
        total_script_chars = devanagari_count + kannada_count + latin_count

        # 1. Native Devanagari script detection
        if devanagari_count >= 2 and (total_script_chars == 0 or devanagari_count / total_script_chars >= 0.3):
            return LanguageDetectionResult(
                detected_language=SupportedLanguage.HINDI,
                confidence=0.98,
                is_uncertain=False,
                method="script_devanagari",
            )

        # 2. Native Kannada script detection
        if kannada_count >= 2 and (total_script_chars == 0 or kannada_count / total_script_chars >= 0.3):
            return LanguageDetectionResult(
                detected_language=SupportedLanguage.KANNADA,
                confidence=0.98,
                is_uncertain=False,
                method="script_kannada",
            )

        # 3. Latin script token analysis
        if latin_count >= 2:
            # Tokenize lowercased words
            words = re.findall(r"\b[a-zA-Z]+\b", clean.lower())
            if not words:
                return LanguageDetectionResult(
                    detected_language=None,
                    confidence=0.0,
                    is_uncertain=True,
                    method="uncertain",
                )

            words_set = set(words)
            hi_matches = len(words_set & _HINDI_TRANSLIT_KEYWORDS)
            kn_matches = len(words_set & _KANNADA_TRANSLIT_KEYWORDS)
            en_matches = len(words_set & _ENGLISH_KEYWORDS)

            # Check transliterated Indic first if distinct Indic tokens are present
            if hi_matches > 0 and hi_matches >= en_matches and hi_matches >= kn_matches:
                return LanguageDetectionResult(
                    detected_language=SupportedLanguage.HINDI,
                    confidence=min(0.90, 0.65 + hi_matches * 0.1),
                    is_uncertain=False,
                    method="latin_transliterated_hindi",
                )

            if kn_matches > 0 and kn_matches >= en_matches and kn_matches >= hi_matches:
                return LanguageDetectionResult(
                    detected_language=SupportedLanguage.KANNADA,
                    confidence=min(0.90, 0.65 + kn_matches * 0.1),
                    is_uncertain=False,
                    method="latin_transliterated_kannada",
                )

            if en_matches > 0:
                return LanguageDetectionResult(
                    detected_language=SupportedLanguage.ENGLISH,
                    confidence=min(0.95, 0.70 + en_matches * 0.08),
                    is_uncertain=False,
                    method="latin_english",
                )

            # General English vocabulary heuristic: if 3+ standard English-like words without non-English markers
            if len(words) >= 3 and not (hi_matches or kn_matches):
                # Mild English assumption for Latin text
                return LanguageDetectionResult(
                    detected_language=SupportedLanguage.ENGLISH,
                    confidence=0.60,
                    is_uncertain=False,
                    method="latin_general_fallback",
                )

        # 4. Uncertain / ambiguous (e.g. only numbers, symbols, unidentifiable text)
        return LanguageDetectionResult(
            detected_language=None,
            confidence=0.10,
            is_uncertain=True,
            method="uncertain",
        )

    def resolve_language(
        self,
        query: str,
        explicit_language: Optional[SupportedLanguage] = None,
    ) -> LanguageResolution:
        """
        Resolve the effective language for a query following NyayaSetu priority rules:
        
        1. An explicit user-selected language TAKES PRIORITY over automatic detection.
        2. If the user has not selected a language (None or 'auto'), detect English, Hindi, or Kannada.
        3. If detection is uncertain, return is_uncertain=True so the caller can prompt the user.
        """
        # Rule 1: Explicit user selection takes precedence
        if explicit_language is not None:
            detection = self.detect_language(query)
            return LanguageResolution(
                language=explicit_language,
                is_explicit=True,
                is_uncertain=False,
                detection=detection,
            )

        # Rule 2: Automatic detection
        detection = self.detect_language(query)
        if detection.is_uncertain or detection.detected_language is None:
            # Fallback to English as safe carrier, but with is_uncertain=True
            return LanguageResolution(
                language=SupportedLanguage.ENGLISH,
                is_explicit=False,
                is_uncertain=True,
                detection=detection,
            )

        return LanguageResolution(
            language=detection.detected_language,
            is_explicit=False,
            is_uncertain=False,
            detection=detection,
        )


# Singleton
language_detector = LanguageDetector()
