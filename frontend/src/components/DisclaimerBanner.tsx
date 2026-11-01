import React from 'react';
import { AlertTriangle } from 'lucide-react';
import type { Language } from '../types';
import { TRANSLATIONS } from '../constants/translations';

interface DisclaimerBannerProps {
  currentLanguage: Language;
}

export const DisclaimerBanner: React.FC<DisclaimerBannerProps> = ({ currentLanguage }) => {
  const t = TRANSLATIONS[currentLanguage];

  return (
    <div className="w-full bg-slate-100/90 border-t border-slate-200/80 py-2.5 px-4">
      <div className="max-w-6xl mx-auto flex items-center justify-center gap-2 text-center">
        <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0 hidden sm:inline" />
        <p className="text-[11px] text-slate-600 leading-snug">
          {t.disclaimerText}
        </p>
      </div>
    </div>
  );
};
