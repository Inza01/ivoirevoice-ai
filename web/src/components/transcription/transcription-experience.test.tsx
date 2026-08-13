import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import axe from "axe-core";
import { afterEach, vi } from "vitest";

import {
  TranscriptionExperience,
  type TranscriptionApi,
} from "@/components/transcription/transcription-experience";
import { I18nProvider } from "@/i18n/provider";
import { ApiClientError } from "@/lib/api/client";
import type { TranscriptionResponse } from "@/lib/api/contracts";

const languages = {
  languages: [
    {
      code: "fr" as const,
      name: "Français",
      asr: "experimental" as const,
      learning: "coming_soon" as const,
      translation_targets: {},
    },
    {
      code: "en" as const,
      name: "English",
      asr: "experimental" as const,
      learning: "coming_soon" as const,
      translation_targets: {},
    },
    {
      code: "dyu" as const,
      name: "Dioula",
      asr: "experimental" as const,
      learning: "coming_soon" as const,
      translation_targets: {},
    },
  ],
};

const models = {
  models: [
    {
      id: "whisper_tiny_baseline",
      display_name: "Whisper Tiny — Baseline",
      status: "experimental" as const,
      supported_languages: ["fr", "en", "dyu"] as const,
    },
    {
      id: "whisper_small_baseline",
      display_name: "Whisper Small — Baseline",
      status: "experimental" as const,
      supported_languages: ["fr", "en", "dyu"] as const,
    },
    {
      id: "whisper_tiny_dioula_final",
      display_name: "Whisper Tiny — Dioula Final",
      status: "experimental" as const,
      supported_languages: ["dyu"] as const,
    },
  ],
};

const completed: TranscriptionResponse = {
  audio_duration_seconds: 5.42,
  id: "request_123",
  language: "dyu",
  model_id: "whisper_tiny_baseline",
  processing_time_seconds: 0.73,
  rtf: 0.13,
  status: "completed",
  text: "An bɛ taa.",
};

afterEach(() => {
  vi.restoreAllMocks();
  Reflect.deleteProperty(navigator, "clipboard");
});

function client(overrides: Partial<TranscriptionApi> = {}): TranscriptionApi {
  return {
    createTranscription: vi.fn(async () => completed),
    listLanguages: vi.fn(async () => languages),
    listModels: vi.fn(async () => models),
    ...overrides,
  };
}

function renderExperience(api = client()) {
  return render(
    <I18nProvider initialLocale="fr">
      <TranscriptionExperience client={api} />
    </I18nProvider>,
  );
}

async function selectAudio() {
  const input = await screen.findByLabelText("Choisir un fichier");
  const file = new File(["synthetic"], "private-name.wav", { type: "audio/wav" });
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

describe("TranscriptionExperience", () => {
  it("discovers runtime capabilities and filters models by selected language", async () => {
    renderExperience();

    const language = await screen.findByLabelText("Langue de l'audio");
    const model = screen.getByLabelText("Modèle de transcription");
    expect(language).toHaveValue("dyu");
    expect(model).toHaveValue("whisper_tiny_baseline");
    expect(screen.getByRole("option", { name: "Whisper Tiny — Dioula Final" })).toBeInTheDocument();

    fireEvent.change(language, { target: { value: "fr" } });

    expect(model).toHaveValue("whisper_tiny_baseline");
    expect(screen.queryByRole("option", { name: "Whisper Tiny — Dioula Final" })).toBeNull();
    expect(screen.getByRole("option", { name: /Auto — Expérimental/ })).toBeDisabled();
  });

  it("submits one generic-name upload and renders the completed public result", async () => {
    const createTranscription = vi.fn<TranscriptionApi["createTranscription"]>(
      async () => completed,
    );
    renderExperience(client({ createTranscription }));
    const file = await selectAudio();

    fireEvent.click(screen.getByRole("button", { name: "Transcrire" }));

    expect(await screen.findByText("An bɛ taa.")).toBeInTheDocument();
    expect(screen.getByText("Transcription terminée")).toBeInTheDocument();
    expect(screen.getAllByText("Whisper Tiny — Baseline")).toHaveLength(2);
    expect(screen.getByText("5,42 s")).toBeInTheDocument();
    expect(createTranscription).toHaveBeenCalledOnce();
    expect(createTranscription.mock.calls[0]?.[0]).toMatchObject({
      audio: file,
      filename: "audio-upload.wav",
      language: "dyu",
      modelId: "whisper_tiny_baseline",
    });
    expect(createTranscription.mock.calls[0]?.[0].signal).toBeInstanceOf(AbortSignal);
  });

  it("exposes a non-duplicating processing state", async () => {
    let resolve!: (value: TranscriptionResponse) => void;
    const pending = new Promise<TranscriptionResponse>((done) => {
      resolve = done;
    });
    const createTranscription = vi.fn<TranscriptionApi["createTranscription"]>(() => pending);
    renderExperience(client({ createTranscription }));
    await selectAudio();

    fireEvent.click(screen.getByRole("button", { name: "Transcrire" }));

    expect(screen.getByRole("status")).toHaveTextContent("Transcription en cours");
    expect(screen.getByRole("button", { name: "Chargement en cours" })).toBeDisabled();
    resolve(completed);
    expect(await screen.findByText("An bɛ taa.")).toBeInTheDocument();
    expect(createTranscription).toHaveBeenCalledOnce();
  });

  it("maps a safe API code without exposing an upstream message or path", async () => {
    const failure = new ApiClientError("model_unavailable", 503, true);
    renderExperience(client({ createTranscription: vi.fn(async () => Promise.reject(failure)) }));
    await selectAudio();

    fireEvent.click(screen.getByRole("button", { name: "Transcrire" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Le modèle sélectionné est momentanément indisponible.");
    expect(alert).not.toHaveTextContent("/home/");
    expect(screen.getByRole("button", { name: "Réessayer" })).toBeEnabled();
  });

  it("copies, exposes reviewed downloads, keeps translation disabled, and clears", async () => {
    const writeText = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    renderExperience();
    await selectAudio();
    fireEvent.click(screen.getByRole("button", { name: "Transcrire" }));
    await screen.findByText("An bɛ taa.");

    fireEvent.click(screen.getByRole("button", { name: "Copier" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("An bɛ taa."));
    expect(screen.getByText("Texte copié.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Télécharger TXT" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Télécharger JSON" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Traduire cette transcription/ })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Effacer" }));
    expect(screen.queryByText("An bɛ taa.")).toBeNull();
    expect(screen.getByText("Votre transcription apparaîtra ici")).toBeInTheDocument();
  });

  it("supports a privacy-safe discovery retry", async () => {
    const listModels = vi
      .fn<TranscriptionApi["listModels"]>()
      .mockRejectedValueOnce(new Error("private upstream failure"))
      .mockResolvedValueOnce(models);
    renderExperience(client({ listModels }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Impossible de charger les capacités",
    );
    fireEvent.click(screen.getByRole("button", { name: "Réessayer" }));

    expect(await screen.findByLabelText("Modèle de transcription")).toBeEnabled();
    expect(listModels).toHaveBeenCalledTimes(2);
  });

  it("has no detectable structural WCAG violations at mobile width", async () => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 320 });
    const { container } = renderExperience();
    await screen.findByLabelText("Modèle de transcription");

    const results = await axe.run(container, {
      rules: { "color-contrast": { enabled: false } },
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"] },
    });

    expect(results.violations).toEqual([]);
    expect(screen.getByRole("button", { name: "Transcrire" })).toBeInTheDocument();
  });
});
