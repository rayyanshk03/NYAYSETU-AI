import type { BackendChatResponse, BackendRetrievalResponse, HealthStatus, Language } from '../types';

// Retrieve backend URL from environment with fallback
const RAW_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string) || 'http://127.0.0.1:8000/api';
// Normalize: ensure it ends with /api without trailing slash
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, '').endsWith('/api')
  ? RAW_BASE_URL.replace(/\/+$/, '')
  : `${RAW_BASE_URL.replace(/\/+$/, '')}/api`;

export class ApiError extends Error {
  statusCode?: number;
  details?: unknown;

  constructor(message: string, statusCode?: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * Fetch health status of the NyayaSetu AI backend service.
 */
export async function checkBackendHealth(): Promise<HealthStatus> {
  const url = `${API_BASE_URL}/health`;
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      throw new ApiError(`Health check failed with HTTP ${response.status}`, response.status);
    }

    const data: HealthStatus = await response.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    const msg = err instanceof Error ? err.message : 'Network error connecting to backend';
    throw new ApiError(`Backend connection failed: ${msg}`);
  }
}

/**
 * Submit user legal inquiry to backend chat endpoint.
 */
export async function sendChatMessage(
  message: string,
  language: Language
): Promise<BackendChatResponse> {
  const trimmed = message.trim();
  if (!trimmed) {
    throw new ApiError('Message cannot be empty', 400);
  }

  const url = `${API_BASE_URL}/chat`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        message: trimmed,
        language: language,
      }),
    });

    if (!response.ok) {
      let errorDetails: unknown = null;
      try {
        errorDetails = await response.json();
      } catch {
        // Body was not JSON
      }
      throw new ApiError(
        `Server returned ${response.status}: ${response.statusText}`,
        response.status,
        errorDetails
      );
    }

    const data: BackendChatResponse = await response.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    const msg = err instanceof Error ? err.message : 'Network error';
    throw new ApiError(`Failed to send message: ${msg}`);
  }
}

/**
 * Submit user legal query directly to the Step 3 RAG /api/retrieve endpoint.
 * Returns retrieved statutory passages and relevance scores without LLM synthesis.
 */
export async function sendRetrievalQuery(
  query: string,
  language: Language = 'en',
  topK: number = 5
): Promise<BackendRetrievalResponse> {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new ApiError('Query cannot be empty', 400);
  }

  const url = `${API_BASE_URL}/retrieve`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        query: trimmed,
        language: language,
        top_k: topK,
      }),
    });

    if (!response.ok) {
      let errorDetails: unknown = null;
      try {
        errorDetails = await response.json();
      } catch {
        // Body was not JSON
      }
      throw new ApiError(
        `Server returned ${response.status}: ${response.statusText}`,
        response.status,
        errorDetails
      );
    }

    const data: BackendRetrievalResponse = await response.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    const msg = err instanceof Error ? err.message : 'Network error';
    throw new ApiError(`Failed to retrieve passages: ${msg}`);
  }
}

