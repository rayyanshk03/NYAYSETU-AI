import React, { useEffect, useRef, useState } from 'react';
import { Scale, User, AlertCircle, ShieldAlert, ChevronDown, ChevronUp, BookOpen, Cpu, ExternalLink } from 'lucide-react';
import type { ChatMessage, Language, RetrievedPassage, AgentStep } from '../types';
import { TRANSLATIONS } from '../constants/translations';

interface ChatAreaProps {
  messages: ChatMessage[];
  isLoading: boolean;
  currentLanguage: Language;
}

// ── Collapsible Sources Panel ──────────────────────────────────────────────

const SourcesPanel: React.FC<{ passages: RetrievedPassage[]; defaultOpen?: boolean }> = ({
  passages,
  defaultOpen = false,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  if (passages.length === 0) return null;

  return (
    <div className="mt-3 rounded-xl border border-blue-100 overflow-hidden shadow-xs">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-blue-50/90 hover:bg-blue-100/70 transition-colors text-left"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-blue-700 shrink-0" />
          <span className="text-[11px] font-semibold text-blue-900">
            {passages.length} Statutory Provision{passages.length > 1 ? 's' : ''} Retrieved
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-blue-700/80 font-medium">
            {open ? 'Hide details' : 'View passages & scores'}
          </span>
          {open ? (
            <ChevronUp className="w-3.5 h-3.5 text-blue-600" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-blue-600" />
          )}
        </div>
      </button>

      {open && (
        <div className="divide-y divide-blue-50 bg-white">
          {passages.map((p, i) => (
            <div key={p.chunk_id || i} className="px-4 py-3 text-[11px]">
              <div className="flex items-start justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="font-bold text-slate-900 text-[11px]">
                    [{i + 1}]
                  </span>
                  <span className="font-semibold text-blue-950">
                    {p.act_name || p.document_title}
                  </span>
                  {p.section_number && (
                    <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-800 font-mono text-[10px] font-medium">
                      § {p.section_number}
                    </span>
                  )}
                  {p.section_title && (
                    <span className="text-slate-600 font-medium">— {p.section_title}</span>
                  )}
                  {p.page_start && (
                    <span className="text-slate-400 text-[10px]">
                      (Page {p.page_start}{p.page_end && p.page_end !== p.page_start ? `–${p.page_end}` : ''})
                    </span>
                  )}
                </div>
                <div className="shrink-0 flex items-center gap-1.5">
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                      p.score >= 0.6
                        ? 'bg-emerald-100 text-emerald-800'
                        : p.score >= 0.4
                        ? 'bg-amber-100 text-amber-800'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                    title="Cosine similarity score (0.0 to 1.0; does not indicate legal validity probability)"
                  >
                    Similarity: {(p.score * 100).toFixed(1)}%
                  </span>
                  {p.source_url && (
                    <a
                      href={p.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800 p-1 rounded hover:bg-blue-50 transition-colors flex items-center gap-0.5 text-[10px]"
                      title="Open authoritative bare act on India Code"
                    >
                      <span>India Code</span>
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              </div>
              <p className="text-slate-700 leading-relaxed bg-slate-50/60 p-2.5 rounded-lg border border-slate-100 font-serif text-[11px]">
                {p.legal_text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Collapsible Agent Trace Panel ──────────────────────────────────────────

const AgentTracePanel: React.FC<{ steps: AgentStep[] }> = ({ steps }) => {
  const [open, setOpen] = useState(false);

  if (steps.length === 0) return null;

  return (
    <div className="mt-2 rounded-xl border border-violet-100 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-3.5 py-2 bg-violet-50/60 hover:bg-violet-100/50 transition-colors text-left"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-violet-700 shrink-0" />
          <span className="text-[11px] font-semibold text-violet-900">
            Agent Trace ({steps.length} steps)
          </span>
        </div>
        {open ? (
          <ChevronUp className="w-3 h-3 text-violet-600" />
        ) : (
          <ChevronDown className="w-3 h-3 text-violet-600" />
        )}
      </button>

      {open && (
        <div className="px-4 py-3 bg-white space-y-2.5">
          {steps.map((step) => (
            <div key={step.step} className="flex gap-2.5">
              <div className="w-5 h-5 rounded-full bg-violet-100 text-violet-800 text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                {step.step}
              </div>
              <div>
                <p className="text-[11px] font-semibold text-slate-800">{step.name}</p>
                <p className="text-[10px] text-slate-500 mt-0.5">{step.result_summary}</p>
                {step.duration_ms !== null && (
                  <p className="text-[10px] text-slate-400 mt-0.5">
                    {step.duration_ms.toFixed(0)} ms
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── Main ChatArea ──────────────────────────────────────────────────────────

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isLoading,
  currentLanguage,
}) => {
  const t = TRANSLATIONS[currentLanguage];
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const renderFormattedText = (text: string) => {
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return <div key={idx} className="h-2" />;
      }

      // Blockquotes: lines starting with >
      if (trimmed.startsWith('>')) {
        const quoteContent = trimmed.replace(/^>\s*/, '').replace(/^"|"$/g, '');
        return (
          <blockquote
            key={idx}
            className="my-2 border-l-3 border-blue-600 bg-blue-50/70 px-3.5 py-2 text-xs italic text-blue-950 rounded-r-md"
          >
            "{quoteContent}"
          </blockquote>
        );
      }

      // Bullet points: lines starting with • or - or *
      if (trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        const bulletText = trimmed.replace(/^[•\-*]\s*/, '');
        return (
          <div key={idx} className="flex items-start gap-2 text-xs text-slate-700 my-1 pl-1">
            <span className="text-blue-600 font-bold">•</span>
            <span
              dangerouslySetInnerHTML={{
                __html: formatInline(bulletText),
              }}
            />
          </div>
        );
      }

      // Regular line with bold formatting
      return (
        <p
          key={idx}
          className="text-xs text-slate-800 leading-relaxed my-1"
          dangerouslySetInnerHTML={{
            __html: formatInline(line),
          }}
        />
      );
    });
  };

  const formatInline = (str: string) => {
    return str
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-slate-950">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic text-slate-700">$1</em>');
  };

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-6">
      {messages.length === 0 ? (
        <div className="h-full min-h-[280px] flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-slate-200 rounded-2xl bg-white/50">
          <div className="w-14 h-14 rounded-2xl bg-blue-900/5 border border-blue-900/10 text-blue-900 flex items-center justify-center mb-4 shadow-xs">
            <Scale className="w-7 h-7" />
          </div>
          <h2 className="text-base font-bold text-slate-800 font-heading">
            {t.emptyStateTitle}
          </h2>
          <p className="text-xs text-slate-500 max-w-md mt-1.5 leading-relaxed">
            {t.emptyStateSubtitle}
          </p>
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-[11px] text-slate-500">
            <span className="px-2.5 py-1 bg-slate-100 rounded-md font-medium text-slate-700">
              ⚖️ Maintenance
            </span>
            <span className="px-2.5 py-1 bg-slate-100 rounded-md font-medium text-slate-700">
              📜 Divorce
            </span>
            <span className="px-2.5 py-1 bg-slate-100 rounded-md font-medium text-slate-700">
              🤝 Child Custody
            </span>
            <span className="px-2.5 py-1 bg-slate-100 rounded-md font-medium text-slate-700">
              🛡️ Domestic Violence
            </span>
          </div>
        </div>
      ) : (
        messages.map((msg) => {
          const isUser = msg.sender === 'user';

          if (isUser) {
            return (
              <div key={msg.id} className="flex justify-end gap-2.5 max-w-3xl ml-auto">
                <div className="flex flex-col items-end">
                  <div className="bg-slate-900 text-white px-4 py-3 rounded-2xl rounded-tr-xs shadow-xs text-xs sm:text-sm leading-relaxed max-w-xl">
                    {msg.text}
                  </div>
                  <div className="flex items-center gap-2 mt-1 px-1 text-[10px] text-slate-400">
                    <span>{msg.timestamp}</span>
                    <span>•</span>
                    <span className="uppercase">{msg.language}</span>
                  </div>
                </div>
                <div className="w-8 h-8 rounded-full bg-slate-200 text-slate-700 flex items-center justify-center shrink-0">
                  <User className="w-4 h-4" />
                </div>
              </div>
            );
          }

          // Assistant Response
          return (
            <div key={msg.id} className="flex justify-start gap-3 max-w-3xl mr-auto">
              <div className="w-8 h-8 rounded-full bg-blue-900 text-amber-300 flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                <Scale className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div
                  className={`p-4 rounded-2xl rounded-tl-xs shadow-xs border ${
                    msg.error
                      ? 'bg-rose-50/70 border-rose-200 text-rose-900'
                      : 'bg-white border-slate-200 text-slate-900'
                  }`}
                >
                  {/* Header info */}
                  <div className="flex items-center justify-between gap-2 pb-2.5 mb-2.5 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-slate-900 font-heading">
                        NyayaSetu AI
                      </span>
                      {msg.evidenceStatus === 'emergency' && (
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-rose-100 text-rose-900 border border-rose-300 animate-pulse flex items-center gap-1">
                          <span>🚨 Immediate Safety Priority</span>
                        </span>
                      )}
                      {msg.evidenceStatus === 'insufficient' && (
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 border border-amber-300">
                          ⚠️ Insufficient Evidence
                        </span>
                      )}
                      {msg.evidenceStatus === 'empty_corpus' && (
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-800 border border-slate-300">
                          📁 Knowledge Base Unindexed
                        </span>
                      )}
                      {msg.evidenceStatus === 'sufficient' && (
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-900 border border-emerald-300 flex items-center gap-1">
                          <span>✓ Evidence Grounded</span>
                        </span>
                      )}
                      {msg.isRetrievalOnly && (
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-900 border border-emerald-200">
                          Step 3: RAG Retrieval
                        </span>
                      )}
                      {msg.isPlaceholder && !msg.evidenceStatus && (
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-900 border border-amber-200">
                          Placeholder
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] text-slate-400">{msg.timestamp}</span>
                  </div>

                  {/* Verified Citations Summary Bar if available */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div className="mb-3 p-2 bg-slate-50 border border-slate-200 rounded-lg flex items-center gap-2 flex-wrap text-[10px]">
                      <span className="font-semibold text-slate-700">Verified Evidence:</span>
                      {msg.citations.map((c) => (
                        <a
                          key={c.chunk_id || c.index}
                          href={c.source_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-2 py-0.5 bg-white border border-blue-200 text-blue-900 hover:text-blue-950 hover:border-blue-400 rounded-md flex items-center gap-1 font-medium transition-colors shadow-2xs"
                        >
                          <span>[{c.index}] {c.act_name} {c.section_number ? `§${c.section_number}` : ''}</span>
                          <ExternalLink className="w-2.5 h-2.5 text-blue-600" />
                        </a>
                      ))}
                    </div>
                  )}

                  {/* Body Content */}
                  <div className="prose-xs text-slate-800 space-y-1">
                    {renderFormattedText(msg.text)}
                  </div>

                  {/* Insufficient Evidence Alert Box */}
                  {msg.insufficientEvidence && (
                    <div className="mt-3 p-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-2.5">
                      <AlertCircle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                      <div>
                        <p className="font-semibold text-amber-950 text-[11px]">
                          No Relevant Statutory Evidence Found
                        </p>
                        <p className="text-[11px] text-amber-900 mt-0.5 leading-relaxed">
                          {msg.evidenceNote ||
                            'The indexed corpus does not contain provisions satisfying the minimum semantic relevance threshold for this query, or the corpus has not yet been indexed.'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Retrieved Sources Panel */}
                  {msg.retrievedPassages && msg.retrievedPassages.length > 0 && (
                    <SourcesPanel
                      passages={msg.retrievedPassages}
                      defaultOpen={msg.isRetrievalOnly}
                    />
                  )}

                  {/* Agent Trace Panel */}
                  {msg.agentSteps && msg.agentSteps.length > 0 && (
                    <AgentTracePanel steps={msg.agentSteps} />
                  )}

                  {/* Legal Disclaimer Box if present */}
                  {msg.disclaimer && !msg.error && (
                    <div className="mt-3.5 pt-3 border-t border-slate-100 flex items-start gap-2 bg-slate-50/70 p-2.5 rounded-lg border border-slate-100">
                      <ShieldAlert className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
                      <p className="text-[11px] text-slate-600 leading-normal">
                        {msg.disclaimer}
                      </p>
                    </div>
                  )}

                  {msg.error && (
                    <div className="mt-2 flex items-center gap-1.5 text-xs text-rose-700 font-medium">
                      <AlertCircle className="w-4 h-4" />
                      <span>{t.errorSending}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })
      )}

      {/* Loading state indicator */}
      {isLoading && (
        <div className="flex justify-start gap-3 max-w-xl mr-auto animate-pulse">
          <div className="w-8 h-8 rounded-full bg-blue-900/80 text-amber-300 flex items-center justify-center shrink-0">
            <Scale className="w-4 h-4 animate-spin" />
          </div>
          <div className="p-3.5 rounded-2xl rounded-tl-xs bg-white border border-slate-200 shadow-xs flex items-center gap-3">
            <div className="flex space-x-1.5">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
            </div>
            <span className="text-xs text-slate-600 font-medium">
              {t.loadingMessage}
            </span>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};
