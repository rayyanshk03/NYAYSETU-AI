"""Chat service: routes requests through the agentic pipeline.

In Step 4 (Agentic AI Layer), this service delegates to the LegalAgent
orchestrator which runs: Query Planning → Legal Retrieval → Answer Synthesis.

The placeholder path from Step 1 is retained as a named fallback so tests
that verify the placeholder schema can still call it directly.
"""

from typing import Optional, TYPE_CHECKING
from app.schemas.chat import ChatRequest, ChatResponse, SupportedLanguage

if TYPE_CHECKING:
    from app.agent.orchestrator import LegalAgent


class ChatService:
    """Service routing legal queries through the agentic pipeline."""

    def __init__(self, agent: Optional["LegalAgent"] = None):
        self._agent = agent

    @property
    def agent(self) -> "LegalAgent":
        if self._agent is None:
            from app.agent.orchestrator import agent as default_agent
            self._agent = default_agent
        return self._agent

    # ------------------------------------------------------------------
    # Primary method: agentic pipeline
    # ------------------------------------------------------------------

    def process_request(self, request: ChatRequest) -> ChatResponse:
        """Run the full agentic pipeline (plan → retrieve → synthesize)."""
        return self.agent.run(request)

    # ------------------------------------------------------------------
    # Legacy: Step-1 placeholder (retained for existing tests)
    # ------------------------------------------------------------------

    def generate_placeholder_response(self, request: ChatRequest) -> ChatResponse:
        """
        Return a development placeholder without invoking the agent.
        Preserved so Step 1 / Step 2 tests that specifically test the
        placeholder path continue to work without modification.
        """
        lang = request.language
        user_message = request.message

        if lang == SupportedLanguage.HINDI:
            reply_text = (
                f"**[न्यायसेतु एआई (NyayaSetu AI) — चरण 1 विकास पूर्वावलोकन]**\n\n"
                f"आपका पारिवारिक कानून संबंधी प्रश्न सफलतापूर्वक प्राप्त हो गया है:\n"
                f"> *\"{user_message}\"*\n\n"
                f"**सिस्टम स्थिति:**\n"
                f"• फ्रंटएंड एवं बैकएंड संचार सुरक्षित रूप से स्थापित हो चुका है।\n"
                f"• चयनित भाषा: **हिंदी (Hindi)**\n"
                f"• अगले चरणों में: एजेंटिक RAG, वाक पहचान जोड़ा जाएगा।\n\n"
                f"*(सूचना: यह कानूनी सलाह नहीं है।)*"
            )
        elif lang == SupportedLanguage.KANNADA:
            reply_text = (
                f"**[ನ್ಯಾಯಸೇತು ಎಐ (NyayaSetu AI) — ಹಂತ 1 ಅಭಿವೃದ್ಧಿ ಮುನ್ನೋಟ]**\n\n"
                f"ನಿಮ್ಮ ಕೌಟುಂಬಿಕ ಕಾನೂನು ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆ ಯಶಸ್ವಿಯಾಗಿ ಸ್ವೀಕರಿಸಲಾಗಿದೆ:\n"
                f"> *\"{user_message}\"*\n\n"
                f"*(ಸೂಚನೆ: ಇದು ಕಾನೂನು ಸಲಹೆ ಅಲ್ಲ.)*"
            )
        else:
            reply_text = (
                f"**[NyayaSetu AI — Step 1 Foundation Placeholder]**\n\n"
                f"Your Indian family law inquiry was received successfully:\n"
                f"> *\"{user_message}\"*\n\n"
                f"**System Status:**\n"
                f"• Frontend-to-backend communication pipeline is verified.\n"
                f"• Target Language: **English**\n\n"
                f"*(Notice: This is a placeholder. The agentic pipeline is active in Step 4.)*"
            )

        return ChatResponse(
            received_message=user_message,
            language=lang,
            reply=reply_text,
            is_placeholder=True,
        )


chat_service = ChatService()
