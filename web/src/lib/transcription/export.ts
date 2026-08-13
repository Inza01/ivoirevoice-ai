import type { TranscriptionResponse } from "@/lib/api/contracts";

export const TRANSCRIPTION_TXT_FILENAME = "ivoirevoice-transcription.txt";
export const TRANSCRIPTION_JSON_FILENAME = "ivoirevoice-transcription.json";

export type PublicTranscriptionExport = {
  readonly duration_seconds: number;
  readonly language: TranscriptionResponse["language"];
  readonly model: string;
  readonly text: string;
};

function completedText(result: TranscriptionResponse): string {
  return result.status === "completed" && typeof result.text === "string" ? result.text : "";
}

export function toPublicTranscriptionExport(
  result: TranscriptionResponse,
): PublicTranscriptionExport {
  return {
    language: result.language,
    model: result.model_id,
    text: completedText(result),
    duration_seconds: result.audio_duration_seconds ?? 0,
  };
}

export function serializeTranscriptionTxt(result: TranscriptionResponse): string {
  return `${completedText(result)}\n`;
}

export function serializeTranscriptionJson(result: TranscriptionResponse): string {
  return `${JSON.stringify(toPublicTranscriptionExport(result), null, 2)}\n`;
}

export function downloadText(content: string, filename: string, mimeType: string): void {
  const objectUrl = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const anchor = document.createElement("a");
  try {
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.rel = "noopener";
    anchor.click();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}
