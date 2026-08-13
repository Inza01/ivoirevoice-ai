import type { CapabilityStatus, LanguageCode } from "@/lib/languages/registry";

export type ServiceStatus = "ok" | "degraded" | "unavailable";
export type TranscriptionStatus = "queued" | "processing" | "completed" | "failed";
export type TranslationStatus = "queued" | "processing" | "completed" | "failed";

export interface HealthResponse {
  readonly status: ServiceStatus;
  readonly service: string;
  readonly version: string;
}

export interface PublicModel {
  readonly id: string;
  readonly display_name: string;
  readonly status: CapabilityStatus;
  readonly supported_languages: readonly LanguageCode[];
}

export interface ModelsResponse {
  readonly models: readonly PublicModel[];
}

export interface PublicLanguage {
  readonly code: LanguageCode;
  readonly asr: CapabilityStatus;
  readonly learning: CapabilityStatus;
  readonly translation_targets: Readonly<Partial<Record<LanguageCode, CapabilityStatus>>>;
}

export interface LanguagesResponse {
  readonly languages: readonly PublicLanguage[];
}

export interface TranscriptionResponse {
  readonly id: string;
  readonly status: TranscriptionStatus;
  readonly language: LanguageCode;
  readonly model_id: string;
  readonly text?: string;
  readonly detected_language?: LanguageCode;
  readonly audio_duration_seconds?: number;
  readonly processing_time_seconds?: number;
  readonly rtf?: number;
  readonly expires_at?: string;
}

/** Only the error code crosses the browser trust boundary. */
export interface ApiErrorEnvelope {
  readonly error: {
    readonly code: string;
  };
}

export interface CreateTranscriptionInput {
  readonly audio: Blob;
  readonly filename: string;
  readonly language: LanguageCode;
  readonly modelId?: string;
}

export interface CreateTranslationInput {
  readonly text: string;
  readonly sourceLanguage: LanguageCode;
  readonly targetLanguage: LanguageCode;
}

export interface TranslationResponse {
  readonly id: string;
  readonly status: TranslationStatus;
  readonly source_language: LanguageCode;
  readonly target_language: LanguageCode;
  readonly translated_text?: string;
  readonly provider_id?: string;
  readonly expires_at?: string;
}
