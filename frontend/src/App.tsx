import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ExampleQuestions } from './components/ExampleQuestions';
import { ChatArea } from './components/ChatArea';
import { ChatInput } from './components/ChatInput';
import { VoiceModal } from './components/VoiceModal';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import type { Language, ChatMessage, HealthStatus } from './types';
import { checkBackendHealth, sendChatMessage, sendRetrievalQuery, ApiError } from './services/api';
import { TRANSLATIONS } from './constants/translations';
import { Trash2, AlertCircle, Sparkles, Database } from 'lucide-react';

export function App() {
  const [currentLanguage, setCurrentLanguage] = useState<Language>('en');
  const [inputValue, setInputValue] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [isVoiceModalOpen, setIsVoiceModalOpen] = useState<boolean>(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [isRetrievalMode, setIsRetrievalMode] = useState<boolean>(false);

  const t = TRANSLATIONS[currentLanguage];

  // Check backend health
  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const data = await checkBackendHealth();
      setHealth(data);
      setNetworkError(null);
    } catch (err: unknown) {
      setHealth(null);
      const msg = err instanceof Error ? err.message : 'Backend unreachable';
      setNetworkError(msg);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  // Handle message submission
  const handleSubmitMessage = async (customMessage?: string) => {
    const textToSend = (customMessage !== undefined ? customMessage : inputValue).trim();
    if (!textToSend || isLoading) return;

    const userMessageId = `user_${Date.now()}`;
    const timestampStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newUserMessage: ChatMessage = {
      id: userMessageId,
      sender: 'user',
      text: textToSend,
      timestamp: timestampStr,
      language: currentLanguage,
    };

    // Add user message to state and reset input field
    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue('');
    setIsLoading(true);
    setNetworkError(null);

    try {
      if (isRetrievalMode) {
        // Step 3 Development Retrieval Mode: directly queries /api/retrieve without LLM synthesis
        const response = await sendRetrievalQuery(textToSend, currentLanguage, 5);

        const summaryText = response.insufficient_evidence
          ? (response.evidence_note || 'No relevant legal passages found in the knowledge base.')
          : `**Retrieved ${response.total_results} relevant statutory provision(s)** from the Indian family law corpus (Step 3 RAG Engine):`;

        const assistantMessage: ChatMessage = {
          id: `retrieval_${Date.now()}`,
          sender: 'assistant',
          text: summaryText,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          language: response.language,
          isRetrievalOnly: true,
          insufficientEvidence: response.insufficient_evidence,
          evidenceNote: response.evidence_note,
          disclaimer: response.disclaimer,
          retrievedPassages: response.passages ?? [],
        };

        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        // Step 4 Agentic Pipeline
        const response = await sendChatMessage(textToSend, currentLanguage);

        const assistantMessage: ChatMessage = {
          id: response.message_id || `bot_${Date.now()}`,
          sender: 'assistant',
          text: response.reply,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          language: response.language,
          isPlaceholder: response.is_placeholder,
          evidenceStatus: response.evidence_status,
          citations: response.citations ?? [],
          warnings: response.warnings ?? [],
          disclaimer: response.disclaimer,
          retrievedPassages: response.retrieved_passages ?? [],
          agentSteps: response.agent_steps ?? [],
        };

        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (err: unknown) {
      const errorMsg =
        err instanceof ApiError && err.statusCode
          ? `Backend Error (${err.statusCode}): ${err.message}`
          : err instanceof Error
          ? err.message
          : 'Failed to communicate with backend server';

      const errorAssistantMessage: ChatMessage = {
        id: `err_${Date.now()}`,
        sender: 'assistant',
        text: `Error connecting to NyayaSetu AI backend:\n${errorMsg}\n\nPlease ensure the FastAPI server is running on http://127.0.0.1:8000 and accessible.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        language: currentLanguage,
        error: true,
      };

      setMessages((prev) => [...prev, errorAssistantMessage]);
      setNetworkError(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  // When example question is selected, populate input and automatically submit or allow edit
  const handleSelectExampleQuestion = (questionText: string) => {
    setInputValue(questionText);
    handleSubmitMessage(questionText);
  };

  const handleClearChat = () => {
    setMessages([]);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
      {/* Top Header */}
      <Header
        currentLanguage={currentLanguage}
        onLanguageChange={setCurrentLanguage}
        health={health}
        healthLoading={healthLoading}
        onRefreshHealth={refreshHealth}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-5xl w-full mx-auto flex flex-col px-4 sm:px-6 py-4">
        {/* Network Error Alert Banner if backend is not reachable */}
        {networkError && !health && !healthLoading && (
          <div className="mb-4 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between gap-3 shadow-xs">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>
                Backend is currently offline or unreachable. Start the FastAPI server using{' '}
                <code className="bg-amber-100 px-1.5 py-0.5 rounded font-mono text-amber-950">
                  uvicorn app.main:app --reload
                </code>
              </span>
            </div>
            <button
              onClick={refreshHealth}
              className="px-2.5 py-1 bg-amber-200/70 hover:bg-amber-200 text-amber-900 font-semibold rounded-lg text-[11px] shrink-0 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Mode Selector and Action Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-2 border-b border-slate-200/70">
          <div className="flex items-center gap-1.5 bg-slate-200/60 p-1 rounded-xl">
            <button
              type="button"
              onClick={() => setIsRetrievalMode(true)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                isRetrievalMode
                  ? 'bg-white text-emerald-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Database className="w-3.5 h-3.5 text-emerald-600" />
              <span>RAG Retrieval Mode (Step 3)</span>
            </button>
            <button
              type="button"
              onClick={() => setIsRetrievalMode(false)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                !isRetrievalMode
                  ? 'bg-white text-blue-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5 text-blue-600" />
              <span>Agentic Mode (Step 4)</span>
            </button>
          </div>

          {messages.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-slate-500">
                {messages.length} message{messages.length > 1 ? 's' : ''}
              </span>
              <button
                onClick={handleClearChat}
                className="flex items-center gap-1 text-xs text-slate-500 hover:text-rose-600 px-2 py-1 rounded-md hover:bg-rose-50 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{t.clearChat}</span>
              </button>
            </div>
          )}
        </div>

        {/* Conversation Stream */}
        <div className="flex-1 flex flex-col min-h-0 bg-white/70 rounded-2xl border border-slate-200/80 shadow-xs mb-4 overflow-hidden">
          <ChatArea
            messages={messages}
            isLoading={isLoading}
            currentLanguage={currentLanguage}
          />
        </div>

        {/* Bottom Section: Example Questions & Input Form */}
        <div className="space-y-3 shrink-0">
          {/* Example Questions Section */}
          <ExampleQuestions
            currentLanguage={currentLanguage}
            onSelectQuestion={handleSelectExampleQuestion}
            disabled={isLoading}
          />

          {/* Chat Input Area */}
          <ChatInput
            value={inputValue}
            onChange={setInputValue}
            onSubmit={() => handleSubmitMessage()}
            isLoading={isLoading}
            currentLanguage={currentLanguage}
            onOpenVoiceModal={() => setIsVoiceModalOpen(true)}
          />
        </div>
      </main>

      {/* Visible Legal Disclaimer Footer */}
      <DisclaimerBanner currentLanguage={currentLanguage} />

      {/* Voice Input Placeholder Modal */}
      <VoiceModal
        isOpen={isVoiceModalOpen}
        onClose={() => setIsVoiceModalOpen(false)}
        currentLanguage={currentLanguage}
      />
    </div>
  );
}

export default App;
