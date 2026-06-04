/**
 * Typed client for the RAG backend.
 *
 * One module owns:
 *  - The request/response types that mirror the FastAPI Pydantic models.
 *  - The three network calls — askQuestion, uploadDocuments, evaluateAnswer.
 *  - The base URL resolution (VITE_API_URL with a localhost fallback).
 *  - The error translation: every non-OK response surfaces as an Error
 *    whose message is safe to render. /evaluate's 503 path raises a
 *    distinct EvaluatorUnavailableError so the UI can branch on it.
 *
 * Components never call `fetch` directly. They call into this module
 * and catch the thrown Errors at the boundary.
 */

const API_URL: string =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

// --- Shared types ---------------------------------------------------------

export interface Source {
  chunk_id: string;
  source: string;
  score: number;
  text: string;
}

// --- /ask -----------------------------------------------------------------

export interface AskRequest {
  question: string;
  use_keyword?: boolean;
  use_rerank?: boolean;
  top_k?: number;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
}

export async function askQuestion(req: AskRequest): Promise<AskResponse> {
  const res = await safeFetch(`${API_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as AskResponse;
}

// --- /upload --------------------------------------------------------------

export interface UploadFileResult {
  filename: string;
  chunks: number;
}

export interface SkippedFile {
  filename: string;
  reason: string;
}

export interface UploadResponse {
  files: UploadFileResult[];
  /** Files the backend accepted but couldn't extract text from
   *  (e.g. scanned/image-only PDF). They're reported here instead of
   *  failing the whole batch — the UI shows them separately. */
  skipped: SkippedFile[];
  total_chunks_in_collection: number;
}

export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  // FastAPI expects multipart with each file under the same field name.
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  const res = await safeFetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw await readError(res);
  return (await res.json()) as UploadResponse;
}

// --- /evaluate ------------------------------------------------------------

export interface EvaluateRequest {
  question: string;
  answer: string;
  contexts: string[];
}

export interface EvaluateResponse {
  faithfulness: number;
  answer_relevancy: number;
}

/**
 * Thrown when the backend returns 503 from /evaluate — Ragas is an
 * optional extra in the deploy. The MetricsPanel branches on this
 * specifically to show a friendly "not enabled" message instead of an
 * error banner.
 */
export class EvaluatorUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EvaluatorUnavailableError';
  }
}

export async function evaluateAnswer(
  req: EvaluateRequest,
): Promise<EvaluateResponse> {
  const res = await safeFetch(`${API_URL}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (res.status === 503) {
    throw new EvaluatorUnavailableError(await readDetail(res));
  }
  if (!res.ok) throw await readError(res);
  return (await res.json()) as EvaluateResponse;
}

// --- Internals ------------------------------------------------------------

/**
 * fetch() throws on network failure (CORS, DNS, offline). Catch it here
 * and re-raise as a normal Error so callers only have to handle one
 * exception type.
 */
async function safeFetch(input: RequestInfo, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (err) {
    const reason =
      err instanceof Error ? err.message : 'unable to reach the API';
    throw new Error(`Falha de rede: ${reason}`);
  }
}

async function readDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as unknown;
    if (
      typeof body === 'object' &&
      body !== null &&
      'detail' in body &&
      typeof (body as { detail: unknown }).detail === 'string'
    ) {
      return (body as { detail: string }).detail;
    }
  } catch {
    /* fall through */
  }
  return res.statusText;
}

async function readError(res: Response): Promise<Error> {
  const detail = await readDetail(res);
  return new Error(`HTTP ${res.status}: ${detail}`);
}
