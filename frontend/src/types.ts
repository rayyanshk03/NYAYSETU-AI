export type Language = 'en' | 'hi' | 'kn';

export interface LanguageOption {
  code: Language;
  label: string;
  nativeLabel: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  language: Language;
  isPlaceholder?: boolean;
  isRetrievalOnly?: boolean;
  insufficientEvidence?: boolean;
  evidenceNote?: string | null;
  error?: boolean;
  disclaimer?: string;
  evidenceStatus?: string;
  citations?: Citation[];
  warnings?: string[];
  retrievedPassages?: RetrievedPassage[];
  agentSteps?: AgentStep[];
}

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  act_name: string;
  section_number: string | null;
  section_title: string | null;
  page_number?: number;
  source_url: string;
  similarity_score?: number;
  verified: boolean;
}

export interface RetrievedPassage {
  chunk_id: string;
  document_id: string;
  document_title: string;
  act_name: string | null;
  section_number: string | null;
  section_title: string | null;
  subsection_id: string | null;
  page_start: number;
  page_end: number;
  source_url: string;
  publication_date: string | null;
  legal_text: string;
  score: number;
  is_split_fallback: boolean;
}

export interface AgentStep {
  step: number;
  name: string;
  description: string;
  result_summary: string;
  duration_ms: number | null;
}

export interface BackendChatResponse {
  status: string;
  message_id: string;
  received_message: string;
  language: Language;
  reply: string;
  is_placeholder: boolean;
  evidence_status?: string;
  citations?: Citation[];
  warnings?: string[];
  timestamp: string;
  disclaimer: string;
  retrieved_passages: RetrievedPassage[];
  agent_steps: AgentStep[];
}

export interface BackendRetrievalResponse {
  status: string;
  query: string;
  language: Language;
  total_results: number;
  insufficient_evidence: boolean;
  evidence_note: string | null;
  passages: RetrievedPassage[];
  timestamp: string;
  disclaimer: string;
}

export interface HealthStatus {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface ExampleQuestion {
  id: string;
  category: 'maintenance' | 'divorce' | 'custody' | 'domestic_violence';
  categoryLabel: string;
  question: string;
}
