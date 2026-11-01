import React from 'react';
import { Landmark, FileText, HeartHandshake, ShieldAlert, ChevronRight } from 'lucide-react';
import type { Language, ExampleQuestion } from '../types';
import { EXAMPLE_QUESTIONS, TRANSLATIONS } from '../constants/translations';

interface ExampleQuestionsProps {
  currentLanguage: Language;
  onSelectQuestion: (questionText: string) => void;
  disabled?: boolean;
}

export const ExampleQuestions: React.FC<ExampleQuestionsProps> = ({
  currentLanguage,
  onSelectQuestion,
  disabled = false,
}) => {
  const t = TRANSLATIONS[currentLanguage];
  const questions = EXAMPLE_QUESTIONS[currentLanguage];

  const getCategoryIcon = (category: ExampleQuestion['category']) => {
    switch (category) {
      case 'maintenance':
        return <Landmark className="w-4 h-4 text-blue-700" />;
      case 'divorce':
        return <FileText className="w-4 h-4 text-indigo-700" />;
      case 'custody':
        return <HeartHandshake className="w-4 h-4 text-teal-700" />;
      case 'domestic_violence':
        return <ShieldAlert className="w-4 h-4 text-rose-700" />;
    }
  };

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          {t.examplesTitle}
        </span>
        <span className="text-[11px] text-slate-400">Click to load query</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        {questions.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelectQuestion(item.question)}
            disabled={disabled}
            className="group flex items-start gap-3 p-3 text-left rounded-xl bg-white border border-slate-200/90 hover:border-blue-400 hover:shadow-xs transition-all text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div className="mt-0.5 p-1.5 rounded-lg bg-slate-50 border border-slate-100 group-hover:bg-blue-50 transition-colors">
              {getCategoryIcon(item.category)}
            </div>
            <div className="flex-1 min-w-0">
              <span className="inline-block text-[11px] font-semibold text-slate-500 uppercase tracking-wide mb-0.5">
                {item.categoryLabel}
              </span>
              <p className="text-xs text-slate-800 font-medium line-clamp-2 leading-relaxed group-hover:text-blue-900 transition-colors">
                {item.question}
              </p>
            </div>
            <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-blue-600 group-hover:translate-x-0.5 transition-all mt-1 shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
};
