import { describe, expect, it } from "vitest";

import { ApiClientError, BACKEND_PROXY_PATH, IvoireVoiceApiClient } from "@/lib/api/client";

describe("IvoireVoiceApiClient", () => {
  it("uses the same-origin backend proxy", async () => {
    let requestedUrl = "";
    const client = new IvoireVoiceApiClient(async (input) => {
      requestedUrl = String(input);
      return new Response(JSON.stringify({ status: "ok" }), {
        headers: { "Content-Type": "application/json" },
      });
    });

    await expect(client.health()).resolves.toMatchObject({ status: "ok" });
    expect(requestedUrl).toBe(`${BACKEND_PROXY_PATH}/api/health`);
  });

  it("builds an audio request without setting an invalid multipart content type", async () => {
    let requestInit: RequestInit | undefined;
    const controller = new AbortController();
    const client = new IvoireVoiceApiClient(async (_input, init) => {
      requestInit = init;
      return new Response(
        JSON.stringify({ id: "request_1", status: "queued", language: "dyu", model_id: "auto" }),
        { headers: { "Content-Type": "application/json" } },
      );
    });

    await client.createTranscription({
      audio: new Blob(["synthetic"], { type: "audio/wav" }),
      filename: "demo.wav",
      language: "dyu",
      modelId: "whisper_tiny_dioula_final",
      signal: controller.signal,
    });

    expect(requestInit?.method).toBe("POST");
    expect(requestInit?.body).toBeInstanceOf(FormData);
    expect(new Headers(requestInit?.headers).has("Content-Type")).toBe(false);
    expect(requestInit?.signal).toBe(controller.signal);
    const form = requestInit?.body as FormData;
    expect(form.get("language")).toBe("dyu");
    expect(form.get("model")).toBe("whisper_tiny_dioula_final");
  });

  it("preserves only a reviewed backend error code", async () => {
    const privateMessage = "failed at /private-data/model.bin";
    const client = new IvoireVoiceApiClient(async () =>
      Response.json(
        { error: { code: "model_unavailable", message: privateMessage } },
        { status: 503 },
      ),
    );

    let observed: unknown;
    try {
      await client.listModels();
    } catch (error) {
      observed = error;
    }

    expect(observed).toMatchObject({ code: "model_unavailable", retryable: true, status: 503 });
    expect(String(observed)).not.toContain(privateMessage);
    expect(String(observed)).not.toContain("/private-data/");
  });

  it("never exposes a backend message or private path", async () => {
    const privateMessage = "Failure for /private-data/audio.wav: sensitive output";
    const client = new IvoireVoiceApiClient(
      async () =>
        new Response(
          JSON.stringify({ error: { code: "internal_failure", message: privateMessage } }),
          { status: 500, headers: { "Content-Type": "application/json" } },
        ),
    );

    let observed: unknown;
    try {
      await client.listModels();
    } catch (error) {
      observed = error;
    }

    expect(observed).toBeInstanceOf(ApiClientError);
    expect(String(observed)).not.toContain(privateMessage);
    expect(String(observed)).not.toContain("/private-data/");
    expect(observed).toMatchObject({
      code: "service_unavailable",
      status: 500,
      retryable: true,
    });
  });

  it("rejects invalid transcription identifiers before a network call", async () => {
    let calls = 0;
    const client = new IvoireVoiceApiClient(async () => {
      calls += 1;
      return new Response("{}");
    });

    await expect(client.getTranscription("../../private")).rejects.toMatchObject({
      code: "invalid_request",
    });
    expect(calls).toBe(0);
  });

  it("defines a translation contract without simulating a provider", async () => {
    let requestedUrl = "";
    let requestInit: RequestInit | undefined;
    const client = new IvoireVoiceApiClient(async (input, init) => {
      requestedUrl = String(input);
      requestInit = init;
      return new Response(
        JSON.stringify({
          id: "translation_1",
          status: "queued",
          source_language: "fr",
          target_language: "dyu",
        }),
        { headers: { "Content-Type": "application/json" } },
      );
    });

    await expect(
      client.createTranslation({
        text: "Texte synthétique",
        sourceLanguage: "fr",
        targetLanguage: "dyu",
      }),
    ).resolves.toMatchObject({ status: "queued" });

    expect(requestedUrl).toBe(`${BACKEND_PROXY_PATH}/api/v1/translations`);
    expect(requestInit).toMatchObject({ method: "POST" });
    expect(JSON.parse(String(requestInit?.body))).toEqual({
      text: "Texte synthétique",
      source_language: "fr",
      target_language: "dyu",
    });
  });

  it("rejects empty or same-language translation requests locally", async () => {
    let calls = 0;
    const client = new IvoireVoiceApiClient(async () => {
      calls += 1;
      return new Response("{}");
    });

    await expect(
      client.createTranslation({ text: " ", sourceLanguage: "fr", targetLanguage: "dyu" }),
    ).rejects.toMatchObject({ code: "invalid_request" });
    await expect(
      client.createTranslation({ text: "test", sourceLanguage: "fr", targetLanguage: "fr" }),
    ).rejects.toMatchObject({ code: "invalid_request" });
    expect(calls).toBe(0);
  });

  it("rejects absolute or traversing backend paths", () => {
    const fetcher = async (): Promise<Response> => new Response("{}");
    expect(() => new IvoireVoiceApiClient(fetcher, "https://example.test")).toThrow();
    expect(() => new IvoireVoiceApiClient(fetcher, "/api/../private")).toThrow();
  });

  it.each([
    "//evil.example",
    "/api\\backend",
    "/api/backend?target=private",
    "/api/backend#private",
    "/api//backend",
    "/api/./backend",
    "/api/%2e%2e/private",
    "/api/%2fprivate",
  ])("rejects the ambiguous or cross-origin base path %s", (basePath) => {
    const fetcher = async (): Promise<Response> => new Response("{}");

    expect(() => new IvoireVoiceApiClient(fetcher, basePath)).toThrow(
      "The API base path must be a safe same-origin path.",
    );
  });

  it("normalizes one trailing slash on a safe same-origin base path", async () => {
    let requestedUrl = "";
    const client = new IvoireVoiceApiClient(async (input) => {
      requestedUrl = String(input);
      return new Response(JSON.stringify({ status: "ok" }), {
        headers: { "Content-Type": "application/json" },
      });
    }, "/api/backend/");

    await client.health();

    expect(requestedUrl).toBe("/api/backend/api/health");
  });

  it.each([
    ["health", (client: IvoireVoiceApiClient) => client.health()],
    ["models", (client: IvoireVoiceApiClient) => client.listModels()],
    ["languages", (client: IvoireVoiceApiClient) => client.listLanguages()],
    ["transcription", (client: IvoireVoiceApiClient) => client.getTranscription("request_1")],
    [
      "translation",
      (client: IvoireVoiceApiClient) =>
        client.createTranslation({
          text: "Texte synthétique",
          sourceLanguage: "fr",
          targetLanguage: "dyu",
        }),
    ],
  ])("rejects an invalid successful %s response at the HTTP boundary", async (_name, call) => {
    const privatePayload = {
      path: "/private-data/audio.wav",
      prediction: "sensitive output",
      status: "invented",
    };
    const client = new IvoireVoiceApiClient(
      async () =>
        new Response(JSON.stringify(privatePayload), {
          headers: { "Content-Type": "application/json" },
        }),
    );

    let observed: unknown;
    try {
      await call(client);
    } catch (error) {
      observed = error;
    }

    expect(observed).toMatchObject({
      code: "unexpected_response",
      status: 200,
      retryable: true,
    });
    expect(String(observed)).not.toContain(privatePayload.path);
    expect(String(observed)).not.toContain(privatePayload.prediction);
  });

  it("accepts only validated model and language capability shapes", async () => {
    const responses = [
      {
        models: [
          {
            id: "whisper_tiny",
            display_name: "Whisper Tiny",
            status: "experimental",
            supported_languages: ["fr", "dyu"],
          },
        ],
      },
      {
        languages: [
          {
            code: "dyu",
            name: "Dioula",
            asr: "experimental",
            learning: "coming_soon",
            translation_targets: { fr: "coming_soon" },
          },
        ],
      },
    ];
    const client = new IvoireVoiceApiClient(
      async () =>
        new Response(JSON.stringify(responses.shift()), {
          headers: { "Content-Type": "application/json" },
        }),
    );

    await expect(client.listModels()).resolves.toMatchObject({
      models: [{ id: "whisper_tiny" }],
    });
    await expect(client.listLanguages()).resolves.toMatchObject({
      languages: [{ code: "dyu" }],
    });
  });

  it("requires public metrics and text on a completed transcription", async () => {
    const client = new IvoireVoiceApiClient(async () =>
      Response.json({
        id: "request_1",
        status: "completed",
        language: "dyu",
        model_id: "whisper_tiny_dioula_final",
      }),
    );

    await expect(
      client.createTranscription({
        audio: new Blob(["synthetic"], { type: "audio/wav" }),
        filename: "audio-upload.wav",
        language: "dyu",
        modelId: "whisper_tiny_dioula_final",
      }),
    ).rejects.toMatchObject({ code: "unexpected_response" });
  });
});
