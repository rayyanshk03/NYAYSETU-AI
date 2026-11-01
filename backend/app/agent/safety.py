"""Safety, crisis detection, and emergency helpline routing for NyayaSetu AI.

Implements statutory safety guardrails and immediate danger detection.
When a user describes imminent physical harm, violence, or life threats,
the system prioritizes authoritative government emergency helplines
(112, 181, 1091, 1098, 15100) rather than generic statutory retrieval.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional
from app.schemas.chat import SupportedLanguage


class EmergencyCheckResult(NamedTuple):
    """Result of safety screening on user inquiry."""

    is_emergency: bool
    category: Optional[str] = None
    response_text: Optional[str] = None


# Official, verified government helplines across India (no fabricated numbers)
OFFICIAL_HELPLINES = {
    "national_emergency": {
        "number": "112",
        "name": "National Emergency Response Support System (ERSS)",
        "description": "Immediate police, fire, and medical emergency assistance across all States and UTs in India.",
    },
    "women_helpline": {
        "number": "181",
        "name": "Women Helpline (Ministry of Women and Child Development)",
        "description": "24x7 toll-free helpline for women facing domestic violence, abuse, or distress.",
    },
    "police_women_cell": {
        "number": "1091",
        "name": "Women in Distress Police Helpline",
        "description": "Immediate police response for women facing harassment, assault, or threats.",
    },
    "ncw_helpline": {
        "number": "7827170170",
        "name": "National Commission for Women (NCW) 24x7 Helpline",
        "description": "National helpline for domestic abuse, harassment, and legal aid coordination.",
    },
    "childline": {
        "number": "1098",
        "name": "Childline India",
        "description": "24x7 emergency helpline for children in need of care, protection, or custody crises.",
    },
    "nalsa_legal_aid": {
        "number": "15100",
        "name": "National Legal Services Authority (NALSA)",
        "description": "Toll-free national legal aid helpline providing free legal representation to eligible citizens.",
        "url": "https://nalsa.gov.in/",
    },
}

# Regex patterns detecting immediate physical danger, assault, or life threats
_EMERGENCY_PATTERNS = [
    # English
    r"\b(?:kill(?:ing)?\s+me|threaten(?:ed|ing)?\s+to\s+kill|threat\s+to\s+life)\b",
    r"\b(?:beat(?:ing)?\s+me|physically\s+assault(?:ing|ed)?|hit(?:ting)?\s+me\s+now)\b",
    r"\b(?:immediate\s+danger|life\s+is\s+in\s+danger|save\s+me\s+please)\b",
    r"\b(?:locked\s+in(?:side)?\s+(?:the\s+)?room|held\s+hostage|cannot\s+escape)\b",
    r"\b(?:call\s+police\s+now|emergency\s+help\s+needed|being\s+attacked)\b",
    r"\b(?:suicid(?:e|al)|want\s+to\s+end\s+my\s+life)\b",
    # Hindi
    r"(?:जान\s*से\s*मारने|मारपीट\s*कर\s*रहा|मुझे\s*बचाओ|तुरंत\s*मदद|जान\s*का\s*खतरा|कमरे\s*में\s*बंद|हिंसा\s*हो\s*रही)",
    # Kannada
    r"(?:ಕೊಲ್ಲುವುದಾಗಿ\s*ಬೆದರಿಕೆ|ಹಲ್ಲೆ\s*ಮಾಡುತ್ತಿದ್ದಾರೆ|ತಕ್ಷಣ\s*ಸಹಾಯ|ಪ್ರಾಣಾಪಾಯ|ನನ್ನನ್ನು\s*ಕಾಪಾಡಿ)",
]

_COMPILED_EMERGENCY = [re.compile(p, re.IGNORECASE) for p in _EMERGENCY_PATTERNS]


class SafetyAuditor:
    """Detects immediate danger and attaches official Indian emergency resources."""

    @staticmethod
    def screen_query(query: str, language: SupportedLanguage = SupportedLanguage.ENGLISH) -> EmergencyCheckResult:
        """
        Screen user query for immediate physical danger, violence, or emergency distress.
        """
        clean_text = query.strip()
        if not clean_text:
            return EmergencyCheckResult(is_emergency=False)

        for pattern in _COMPILED_EMERGENCY:
            if pattern.search(clean_text):
                crisis_response = SafetyAuditor._build_emergency_response(language)
                return EmergencyCheckResult(
                    is_emergency=True,
                    category="immediate_danger",
                    response_text=crisis_response,
                )

        return EmergencyCheckResult(is_emergency=False)

    @staticmethod
    def _build_emergency_response(language: SupportedLanguage) -> str:
        """Compose clear emergency advisory with official Indian government helpline numbers."""
        if language == SupportedLanguage.HINDI:
            return (
                "🚨 **तत्काल सुरक्षा चेतावनी एवं आपातकालीन सहायता (Emergency Assistance)**\n\n"
                "यदि आप या आपका कोई परिजन तत्काल शारीरिक खतरे, घरेलू हिंसा या असुरक्षा में हैं, "
                "तो कृपया पहले तुरंत सुरक्षित स्थान पर जाएं और निम्नलिखित **आधिकारिक सरकारी हेल्पलाइन** पर संपर्क करें:\n\n"
                "• **112** — राष्ट्रीय आपातकालीन सेवा (National Emergency - Police / Medical)\n"
                "• **181** — महिला हेल्पलाइन (घरेलू हिंसा एवं संकट सहायता - 24x7 टोल-फ्री)\n"
                "• **1091** — महिला पुलिस हेल्पलाइन (Women Police Distress Cell)\n"
                "• **7827170170** — राष्ट्रीय महिला आयोग (NCW 24x7 Helpline)\n"
                "• **1098** — चाइल्डलाइन (बच्चों की सुरक्षा एवं सहायता)\n"
                "• **15100** — राष्ट्रीय विधिक सेवा प्राधिकरण (NALSA - मुफ्त कानूनी सहायता)\n\n"
                "⚠️ *न्यायसेतु एआई (NyayaSetu AI) केवल सामान्य सूचनात्मक प्रणाली है और आपातकालीन पुलिस सेवा का स्थान नहीं ले सकती। कृपया तुरंत 112 या 181 पर कॉल करें।*"
            )
        elif language == SupportedLanguage.KANNADA:
            return (
                "🚨 **ತುರ್ತು ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆ ಮತ್ತು ಸಹಾಯ (Emergency Assistance)**\n\n"
                "ನೀವು ಅಥವಾ ನಿಮ್ಮ ಕುಟುಂಬದವರು ತಕ್ಷಣದ ದೈಹಿಕ ಅಪಾಯ ಅಥವಾ ಕೌಟುಂಬಿಕ ಹಿಂಸಾಚಾರದಲ್ಲಿದ್ದರೆ, "
                "ದಯವಿಟ್ಟು ಮೊದಲು ಸುರಕ್ಷಿತ ಸ್ಥಳಕ್ಕೆ ತೆರಳಿ ಮತ್ತು ಈ ಕೆಳಗಿನ **ಅಧಿಕೃತ ಸರ್ಕಾರಿ ಸಹಾಯವಾಣಿಗಳಿಗೆ** ಸಂಪರ್ಕಿಸಿ:\n\n"
                "• **112** — ರಾಷ್ಟ್ರೀಯ ತುರ್ತು ಸಹಾಯವಾಣಿ (ಪೊಲೀಸ್ / ಆಂಬ್ಯುಲೆನ್ಸ್ - 24x7)\n"
                "• **181** — ಮಹಿಳಾ ಸಹಾಯವಾಣಿ (ಕೌಟುಂಬಿಕ ದೌರ್ಜನ್ಯ ಮತ್ತು ರಕ್ಷಣೆ - ಟೋಲ್-ಫ್ರೀ)\n"
                "• **1091** — ಮಹಿಳಾ ಪೊಲೀಸ್ ಸಹಾಯವಾಣಿ\n"
                "• **7827170170** — ರಾಷ್ಟ್ರೀಯ ಮಹಿಳಾ ಆಯೋಗ (NCW 24x7)\n"
                "• **1098** — ಮಕ್ಕಳ ಸಹಾಯವಾಣಿ (Childline)\n"
                "• **15100** — ರಾಷ್ಟ್ರೀಯ ಕಾನೂನು ಸೇವೆಗಳ ಪ್ರಾಧಿಕಾರ (NALSA - ಉಚಿತ ಕಾನೂನು ನೆರವು)\n\n"
                "⚠️ *ನ್ಯಾಯಸೇತು ಎಐ ಕೇವಲ ಮಾಹಿತಿ ನೀಡುವ ವ್ಯವಸ್ಥೆಯಾಗಿದ್ದು, ತುರ್ತು ಪೊಲೀಸ್ ರಕ್ಷಣೆಗೆ ಪರ್ಯಾಯವಲ್ಲ. ತಕ್ಷಣ 112 ಅಥವಾ 181 ಕ್ಕೆ ಕರೆ ಮಾಡಿ.*"
            )
        else:
            return (
                "🚨 **IMMEDIATE SAFETY ADVISORY & EMERGENCY RESOURCES**\n\n"
                "If you or someone in your care is in immediate physical danger, facing violence, "
                "or unable to safely leave your premises, **prioritize your immediate physical safety first** "
                "and reach out to verified Indian official emergency services:\n\n"
                "• **112** — National Emergency Response Support System (Police / Medical / Rescue — 24x7)\n"
                "• **181** — Women Helpline (Ministry of WCD — 24x7 Toll-Free for domestic abuse & distress)\n"
                "• **1091** — Women in Distress Police Helpline (Immediate police dispatch)\n"
                "• **7827170170** — National Commission for Women (NCW 24x7 Crisis Support)\n"
                "• **1098** — Childline India (Immediate crisis support for minors & children)\n"
                "• **15100** — National Legal Services Authority ([NALSA](https://nalsa.gov.in/) — Free Legal Aid)\n\n"
                "⚠️ *NyayaSetu AI provides general legal information only and cannot dispatch police or emergency rescue. Please dial 112 or 181 immediately if you are in danger.*"
            )


safety_auditor = SafetyAuditor()
