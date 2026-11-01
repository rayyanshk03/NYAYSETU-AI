import React from 'react';
import { Mic, X, CheckCircle2 } from 'lucide-react';
import type { Language } from '../types';
import { TRANSLATIONS } from '../constants/translations';

interface VoiceModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentLanguage: Language;
}

export const VoiceModal: React.FC<VoiceModalProps> = ({
  isOpen,
  onClose,
  currentLanguage,
}) => {
  if (!isOpen) return null;
  const t = TRANSLATIONS[currentLanguage];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-in fade-in">
      <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-xl border border-slate-200 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-700 border border-blue-200 flex items-center justify-center">
            <Mic className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900 font-heading">
              {t.micModalTitle}
            </h3>
            <span className="text-[11px] font-semibold text-amber-800 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-200">
              Planned for Step 3
            </span>
          </div>
        </div>

        <p className="text-xs text-slate-600 leading-relaxed mb-4">
          {t.micModalDesc}
        </p>

        <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3.5 space-y-2 mb-5">
          <div className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider">
            Future Speech Architecture:
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0" />
            <span>Indic speech-to-text models (Whisper / IndicConformer)</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0" />
            <span>Accent & dialect normalization for English, Hindi & Kannada</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0" />
            <span>Natural voice feedback (Text-to-Speech)</span>
          </div>
        </div>

        <button
          onClick={onClose}
          className="w-full py-2.5 px-4 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold shadow-xs transition-colors"
        >
          Understood, Continue Typing
        </button>
      </div>
    </div>
  );
};
