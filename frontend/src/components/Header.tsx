import React from 'react';
import { Scale, Globe, RefreshCw } from 'lucide-react';
import type { Language, HealthStatus } from '../types';
import { LANGUAGES, TRANSLATIONS } from '../constants/translations';

interface HeaderProps {
  currentLanguage: Language;
  onLanguageChange: (lang: Language) => void;
  health: HealthStatus | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentLanguage,
  onLanguageChange,
  health,
  healthLoading,
  onRefreshHealth,
}) => {
  const t = TRANSLATIONS[currentLanguage];

  return (
    <header className="border-b border-slate-200/80 bg-white/95 backdrop-blur-sm sticky top-0 z-40 shadow-xs">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3.5 flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Branding */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-900 text-amber-300 flex items-center justify-center shadow-xs ring-1 ring-blue-950/20">
            <Scale className="w-6 h-6" strokeWidth={2.2} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-slate-900 font-heading">
                {t.appName}
              </h1>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-800 border border-blue-200/80">
                {t.stepBadge}
              </span>
            </div>
            <p className="text-xs text-slate-500 line-clamp-1 max-w-xl">
              {t.subTitle}
            </p>
          </div>
        </div>

        {/* Controls: Health Status & Language Selector */}
        <div className="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-end">
          {/* Health indicator */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                health
                  ? 'bg-emerald-500 ring-2 ring-emerald-200 animate-pulse'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-slate-600 font-medium">
              {healthLoading
                ? t.backendChecking
                : health
                ? `${t.backendOnline} (v${health.version})`
                : t.backendOffline}
            </span>
            <button
              onClick={onRefreshHealth}
              disabled={healthLoading}
              title="Refresh backend status"
              className="text-slate-400 hover:text-slate-700 transition-colors p-0.5"
            >
              <RefreshCw
                className={`w-3 h-3 ${healthLoading ? 'animate-spin' : ''}`}
              />
            </button>
          </div>

          {/* Language Selector */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-xl border border-slate-200">
            <Globe className="w-4 h-4 text-slate-500 ml-1.5" />
            <div className="flex items-center space-x-1">
              {LANGUAGES.map((lang) => {
                const isActive = currentLanguage === lang.code;
                return (
                  <button
                    key={lang.code}
                    onClick={() => onLanguageChange(lang.code)}
                    className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition-all ${
                      isActive
                        ? 'bg-blue-900 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                    }`}
                    title={`Switch to ${lang.label}`}
                  >
                    <span>{lang.nativeLabel}</span>
                    <span className="ml-1 text-[10px] opacity-70">
                      ({lang.code.toUpperCase()})
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
