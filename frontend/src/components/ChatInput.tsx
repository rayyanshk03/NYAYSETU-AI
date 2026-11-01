import React, { useRef, useEffect } from 'react';
import { Send, Mic } from 'lucide-react';
import type { Language } from '../types';
import { TRANSLATIONS } from '../constants/translations';

interface ChatInputProps {
  value: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  currentLanguage: Language;
  onOpenVoiceModal: () => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  isLoading,
  currentLanguage,
  onOpenVoiceModal,
}) => {
  const t = TRANSLATIONS[currentLanguage];
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isBlank = !value.trim();

  // Adjust textarea height dynamically
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(Math.max(scrollHeight, 84), 200)}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isBlank && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div className="w-full bg-white rounded-2xl border border-slate-200/90 shadow-sm focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-100 transition-all p-3">
      {/* Textarea */}
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={t.inputPlaceholder}
        disabled={isLoading}
        rows={3}
        className="w-full resize-none bg-transparent border-0 outline-none text-slate-800 placeholder-slate-400 text-sm leading-relaxed focus:ring-0 p-1"
      />

      {/* Input Action Bar */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-100 mt-1">
        {/* Left: Microphone button placeholder & hint */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onOpenVoiceModal}
            title={t.micTooltip}
            className="group relative flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-dashed border-slate-300 bg-slate-50 text-slate-600 hover:bg-slate-100 hover:text-slate-900 hover:border-slate-400 transition-colors text-xs"
          >
            <Mic className="w-4 h-4 text-slate-500 group-hover:text-blue-600 transition-colors" />
            <span className="text-[11px] font-medium hidden sm:inline">
              Voice (Step 3)
            </span>
            <span className="text-[9px] uppercase px-1 py-0.2 bg-slate-200 text-slate-600 rounded font-bold">
              Placeholder
            </span>
          </button>
          <span className="text-[11px] text-slate-400 hidden md:inline">
            Press <kbd className="px-1 py-0.5 bg-slate-100 rounded text-[10px] font-mono text-slate-600 border border-slate-200">Enter ↵</kbd> to send
          </span>
        </div>

        {/* Right: Character count & Send button */}
        <div className="flex items-center gap-2">
          {value.length > 0 && (
            <span className="text-[11px] text-slate-400">
              {value.length} chars
            </span>
          )}

          <button
            type="button"
            onClick={onSubmit}
            disabled={isBlank || isLoading}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold shadow-xs transition-all ${
              isBlank || isLoading
                ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
                : 'bg-blue-900 hover:bg-blue-800 text-white shadow-blue-900/10 hover:shadow-sm active:scale-98'
            }`}
          >
            {isLoading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>{t.sendButton}</span>
                <Send className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
