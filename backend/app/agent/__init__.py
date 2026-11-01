"""Agentic AI orchestration layer for NyayaSetu AI.

Provides intent classification, retrieval planning, grounded synthesis,
and citation assembly over the indexed Indian family law corpus.
"""

from .planner import QueryPlanner, QueryPlan, QueryIntent, UserIntentType
from .synthesizer import LegalSynthesizer, SynthesizedResult, synthesizer
from .safety import SafetyAuditor, safety_auditor
from .orchestrator import LegalAgent, agent
from .language_detector import LanguageDetector, language_detector, LanguageDetectionResult, LanguageResolution

__all__ = [
    "QueryPlanner",
    "QueryPlan",
    "QueryIntent",
    "UserIntentType",
    "LegalSynthesizer",
    "SynthesizedResult",
    "synthesizer",
    "SafetyAuditor",
    "safety_auditor",
    "LegalAgent",
    "agent",
    "LanguageDetector",
    "language_detector",
    "LanguageDetectionResult",
    "LanguageResolution",
]
