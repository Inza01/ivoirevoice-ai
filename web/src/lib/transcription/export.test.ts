import {
  serializeTranscriptionJson,
  serializeTranscriptionTxt,
  toPublicTranscriptionExport,
} from "@/lib/transcription/export";

const result = {
  audio_duration_seconds: 5.42,
  id: "request_123",
  language: "dyu" as const,
  model_id: "whisper_tiny_dioula_final",
  processing_time_seconds: 0.73,
  rtf: 0.13,
  status: "completed" as const,
  text: "An bɛ taa.",
};

describe("privacy-safe transcription exports", () => {
  it("exports only the reviewed public JSON fields", () => {
    const exported = toPublicTranscriptionExport(result);

    expect(exported).toEqual({
      duration_seconds: 5.42,
      language: "dyu",
      model: "whisper_tiny_dioula_final",
      text: "An bɛ taa.",
    });
    expect(JSON.stringify(exported)).not.toContain("request_123");
    expect(JSON.stringify(exported)).not.toContain("processing_time_seconds");
    expect(JSON.stringify(exported)).not.toContain("rtf");
  });

  it("serializes TXT and JSON without an uploaded filename or private metadata", () => {
    expect(serializeTranscriptionTxt(result)).toBe("An bɛ taa.\n");
    const json = serializeTranscriptionJson(result);
    expect(JSON.parse(json)).toEqual(toPublicTranscriptionExport(result));
    expect(json).not.toContain("audio.wav");
    expect(json).not.toContain("checkpoint");
  });
});
