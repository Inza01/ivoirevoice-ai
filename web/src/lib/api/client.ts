import type {
  ApiErrorEnvelope,
  CreateTranslationInput,
  CreateTranscriptionInput,
  HealthResponse,
  LanguagesResponse,
  ModelsResponse,
  TranscriptionResponse,
  TranslationResponse,
} from "@/lib/api/contracts";
import {
  CAPABILITY_STATUSES,
  isLanguageCode,
  type CapabilityStatus,
  type LanguageCode,
} from "@/lib/languages/registry";

export const BACKEND_PROXY_PATH = "/api/backend";

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
type ResponseValidator<T> = (value: unknown) => value is T;

export type PublicErrorCode =
  | "incompatible_model_language"
  | "invalid_request"
  | "model_unavailable"
  | "network_error"
  | "not_found"
  | "payload_too_large"
  | "service_unavailable"
  | "transcription_failed"
  | "unknown_language"
  | "unknown_model"
  | "unsupported_audio"
  | "unexpected_response";

const FORWARDED_ERROR_CODES = new Set<PublicErrorCode>([
  "incompatible_model_language",
  "invalid_request",
  "model_unavailable",
  "not_found",
  "payload_too_large",
  "service_unavailable",
  "transcription_failed",
  "unknown_language",
  "unknown_model",
  "unsupported_audio",
]);

export class ApiClientError extends Error {
  readonly code: PublicErrorCode;
  readonly status: number | null;
  readonly retryable: boolean;

  constructor(code: PublicErrorCode, status: number | null, retryable: boolean) {
    // The UI maps this stable code to its active locale. Backend messages are
    // deliberately not propagated across the browser trust boundary.
    super(code);
    this.name = "ApiClientError";
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

function validateBasePath(basePath: string): string {
  const normalized = basePath.replace(/\/$/, "");
  const segments = normalized.slice(1).split("/");
  const hasUnsafeSegment = segments.some(
    (segment) =>
      !segment ||
      segment === "." ||
      segment === ".." ||
      !/^[A-Za-z0-9][A-Za-z0-9._~-]*$/.test(segment),
  );
  if (
    !normalized.startsWith("/") ||
    normalized.startsWith("//") ||
    normalized.includes("\\") ||
    normalized.includes("?") ||
    normalized.includes("#") ||
    hasUnsafeSegment
  ) {
    throw new Error("The API base path must be a safe same-origin path.");
  }
  return normalized;
}

function statusError(status: number, backendCode: string | null): ApiClientError {
  if (backendCode && FORWARDED_ERROR_CODES.has(backendCode as PublicErrorCode)) {
    const code = backendCode as PublicErrorCode;
    return new ApiClientError(code, status, code === "model_unavailable" || status >= 500);
  }
  if (status === 400 || status === 422) {
    return new ApiClientError("invalid_request", status, false);
  }
  if (status === 404) {
    return new ApiClientError("not_found", status, false);
  }
  if (status === 413) {
    return new ApiClientError("payload_too_large", status, false);
  }
  if (status === 415) {
    return new ApiClientError("unsupported_audio", status, false);
  }
  return new ApiClientError("service_unavailable", status, status >= 500);
}

function isErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const error = value.error;
  return (
    typeof error === "object" && error !== null && "code" in error && typeof error.code === "string"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isOptionalString(value: unknown): value is string | undefined {
  return value === undefined || typeof value === "string";
}

function isOptionalNonNegativeNumber(value: unknown): value is number | undefined {
  return value === undefined || (typeof value === "number" && Number.isFinite(value) && value >= 0);
}

function isCapabilityStatus(value: unknown): value is CapabilityStatus {
  return typeof value === "string" && CAPABILITY_STATUSES.some((status) => status === value);
}

function isLanguageCodeArray(value: unknown): value is readonly LanguageCode[] {
  return Array.isArray(value) && value.every((item: unknown) => isLanguageCode(String(item)));
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (!isRecord(value)) return false;
  return value.status === "ok";
}

function isModelsResponse(value: unknown): value is ModelsResponse {
  if (!isRecord(value) || !Array.isArray(value.models)) return false;
  return value.models.every(
    (model: unknown) =>
      isRecord(model) &&
      isNonEmptyString(model.id) &&
      isNonEmptyString(model.display_name) &&
      isCapabilityStatus(model.status) &&
      isLanguageCodeArray(model.supported_languages),
  );
}

function isTranslationTargets(
  value: unknown,
): value is Readonly<Partial<Record<LanguageCode, CapabilityStatus>>> {
  if (!isRecord(value)) return false;
  return Object.entries(value).every(
    ([language, status]) => isLanguageCode(language) && isCapabilityStatus(status),
  );
}

function isLanguagesResponse(value: unknown): value is LanguagesResponse {
  if (!isRecord(value) || !Array.isArray(value.languages)) return false;
  return value.languages.every(
    (language: unknown) =>
      isRecord(language) &&
      isLanguageCode(String(language.code)) &&
      isNonEmptyString(language.name) &&
      isCapabilityStatus(language.asr) &&
      isCapabilityStatus(language.learning) &&
      isTranslationTargets(language.translation_targets),
  );
}

function isTranscriptionResponse(value: unknown): value is TranscriptionResponse {
  if (!isRecord(value)) return false;
  const status = String(value.status);
  const baseIsValid =
    isNonEmptyString(value.id) &&
    /^[A-Za-z0-9_-]{1,128}$/.test(value.id) &&
    ["queued", "processing", "completed", "failed"].includes(status) &&
    isLanguageCode(String(value.language)) &&
    isNonEmptyString(value.model_id) &&
    /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value.model_id) &&
    isOptionalString(value.text) &&
    (value.detected_language === undefined || isLanguageCode(String(value.detected_language))) &&
    isOptionalNonNegativeNumber(value.audio_duration_seconds) &&
    isOptionalNonNegativeNumber(value.processing_time_seconds) &&
    isOptionalNonNegativeNumber(value.rtf) &&
    isOptionalString(value.expires_at);
  if (!baseIsValid) return false;
  if (status === "completed") {
    return (
      typeof value.text === "string" &&
      typeof value.audio_duration_seconds === "number" &&
      value.audio_duration_seconds > 0 &&
      typeof value.processing_time_seconds === "number"
    );
  }
  return true;
}

function isTranslationResponse(value: unknown): value is TranslationResponse {
  if (!isRecord(value)) return false;
  return (
    isNonEmptyString(value.id) &&
    ["queued", "processing", "completed", "failed"].includes(String(value.status)) &&
    isLanguageCode(String(value.source_language)) &&
    isLanguageCode(String(value.target_language)) &&
    value.source_language !== value.target_language &&
    isOptionalString(value.translated_text) &&
    isOptionalString(value.provider_id) &&
    isOptionalString(value.expires_at)
  );
}

async function readErrorCode(response: Response): Promise<string | null> {
  try {
    const payload: unknown = await response.json();
    return isErrorEnvelope(payload) ? payload.error.code : null;
  } catch {
    return null;
  }
}

export class IvoireVoiceApiClient {
  readonly #basePath: string;
  readonly #fetch: FetchLike;

  constructor(
    fetcher: FetchLike = globalThis.fetch.bind(globalThis),
    basePath = BACKEND_PROXY_PATH,
  ) {
    this.#fetch = fetcher;
    this.#basePath = validateBasePath(basePath);
  }

  async #request<T>(path: string, validator: ResponseValidator<T>, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
      response = await this.#fetch(`${this.#basePath}${path}`, {
        ...init,
        headers: {
          Accept: "application/json",
          ...init?.headers,
        },
      });
    } catch {
      throw new ApiClientError("network_error", null, true);
    }

    if (!response.ok) {
      // Consume only the machine-readable code. Backend messages may contain
      // sensitive implementation details and are never surfaced by this client.
      const backendCode = await readErrorCode(response);
      throw statusError(response.status, backendCode);
    }

    try {
      const payload: unknown = await response.json();
      if (!validator(payload)) {
        throw new ApiClientError("unexpected_response", response.status, true);
      }
      return payload;
    } catch {
      throw new ApiClientError("unexpected_response", response.status, true);
    }
  }

  health(): Promise<HealthResponse> {
    return this.#request<HealthResponse>("/api/health", isHealthResponse);
  }

  listModels(signal?: AbortSignal): Promise<ModelsResponse> {
    return this.#request<ModelsResponse>("/api/v1/models", isModelsResponse, { signal });
  }

  listLanguages(signal?: AbortSignal): Promise<LanguagesResponse> {
    return this.#request<LanguagesResponse>("/api/v1/languages", isLanguagesResponse, { signal });
  }

  createTranscription(input: CreateTranscriptionInput): Promise<TranscriptionResponse> {
    const form = new FormData();
    form.append("audio", input.audio, input.filename);
    form.append("language", input.language);
    if (input.modelId) {
      form.append("model", input.modelId);
    }
    return this.#request<TranscriptionResponse>("/api/v1/transcriptions", isTranscriptionResponse, {
      method: "POST",
      body: form,
      signal: input.signal,
    });
  }

  getTranscription(id: string): Promise<TranscriptionResponse> {
    const normalized = id.trim();
    if (!normalized || !/^[A-Za-z0-9_-]+$/.test(normalized)) {
      return Promise.reject(new ApiClientError("invalid_request", null, false));
    }
    return this.#request<TranscriptionResponse>(
      `/api/v1/transcriptions/${encodeURIComponent(normalized)}`,
      isTranscriptionResponse,
    );
  }

  createTranslation(input: CreateTranslationInput): Promise<TranslationResponse> {
    const text = input.text.trim();
    if (!text || input.sourceLanguage === input.targetLanguage) {
      return Promise.reject(new ApiClientError("invalid_request", null, false));
    }

    return this.#request<TranslationResponse>("/api/v1/translations", isTranslationResponse, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        source_language: input.sourceLanguage,
        target_language: input.targetLanguage,
      }),
    });
  }
}

export function createApiClient(fetcher?: FetchLike): IvoireVoiceApiClient {
  return new IvoireVoiceApiClient(fetcher);
}
