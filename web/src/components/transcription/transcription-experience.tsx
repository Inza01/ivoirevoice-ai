"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { AudioUploader, safeAudioUploadFilename } from "@/components/transcription/audio-uploader";
import { TranscriptionResult } from "@/components/transcription/transcription-result";
import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { LanguageSelector } from "@/components/ui/language-selector";
import { useI18n } from "@/i18n/provider";
import type {
  CreateTranscriptionInput,
  LanguagesResponse,
  ModelsResponse,
  PublicLanguage,
  PublicModel,
  TranscriptionResponse,
} from "@/lib/api/contracts";
import {
  ApiClientError,
  createApiClient,
  type IvoireVoiceApiClient,
  type PublicErrorCode,
} from "@/lib/api/client";
import { getLanguageName, type LanguageCode } from "@/lib/languages/registry";

export type TranscriptionApi = Pick<
  IvoireVoiceApiClient,
  "createTranscription" | "listLanguages" | "listModels"
>;

type DiscoveryState = "loading" | "ready" | "error";
export type TranscriptionExperienceState = "idle" | "ready" | "processing" | "success" | "error";

type TranscriptionExperienceProps = {
  client?: TranscriptionApi;
};

function isUsable(status: PublicModel["status"] | PublicLanguage["asr"]): boolean {
  return status === "available" || status === "experimental";
}

function errorCode(error: unknown): PublicErrorCode {
  return error instanceof ApiClientError ? error.code : "unexpected_response";
}

export function TranscriptionExperience({ client }: TranscriptionExperienceProps) {
  const { locale, messages } = useI18n();
  const api = useMemo(() => client ?? createApiClient(), [client]);
  const activeRequest = useRef<AbortController | null>(null);
  const errorRef = useRef<HTMLDivElement>(null);
  const [discoveryAttempt, setDiscoveryAttempt] = useState(0);
  const [discovery, setDiscovery] = useState<DiscoveryState>("loading");
  const [languages, setLanguages] = useState<readonly PublicLanguage[]>([]);
  const [models, setModels] = useState<readonly PublicModel[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<LanguageCode>("dyu");
  const [modelId, setModelId] = useState("");
  const [state, setState] = useState<TranscriptionExperienceState>("idle");
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  const [requestError, setRequestError] = useState<PublicErrorCode | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    void Promise.all([api.listLanguages(controller.signal), api.listModels(controller.signal)])
      .then(([languageResponse, modelResponse]: [LanguagesResponse, ModelsResponse]) => {
        if (!active) return;
        const usableLanguages = languageResponse.languages.filter((item) => isUsable(item.asr));
        const usableModels = modelResponse.models.filter((item) => isUsable(item.status));
        if (!usableLanguages.length || !usableModels.length) throw new Error("empty_discovery");
        const preferredLanguage = usableLanguages.some((item) => item.code === "dyu")
          ? "dyu"
          : usableLanguages[0]!.code;
        const firstCompatible = usableModels.find((item) =>
          item.supported_languages.includes(preferredLanguage),
        );
        setLanguages(usableLanguages);
        setModels(usableModels);
        setLanguage(preferredLanguage);
        setModelId(firstCompatible?.id ?? "");
        setDiscovery("ready");
      })
      .catch(() => {
        if (active && !controller.signal.aborted) setDiscovery("error");
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [api, discoveryAttempt]);

  useEffect(
    () => () => {
      activeRequest.current?.abort();
    },
    [],
  );

  useEffect(() => {
    if (state === "error") errorRef.current?.focus();
  }, [state]);

  const compatibleModels = models.filter((model) => model.supported_languages.includes(language));

  const requestErrorMessage = (() => {
    switch (requestError) {
      case "incompatible_model_language":
        return messages.errors.incompatibleModelLanguage;
      case "model_unavailable":
        return messages.errors.modelUnavailable;
      case "network_error":
        return messages.errors.network;
      case "payload_too_large":
        return messages.errors.fileTooLarge;
      case "transcription_failed":
        return messages.errors.transcriptionFailed;
      case "unknown_language":
        return messages.errors.unknownLanguage;
      case "unknown_model":
      case "not_found":
        return messages.errors.unknownModel;
      case "unsupported_audio":
        return messages.errors.invalidFile;
      case "service_unavailable":
        return messages.errors.unavailable;
      default:
        return messages.errors.generic;
    }
  })();

  const clear = () => {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setFile(null);
    setResult(null);
    setRequestError(null);
    setState("idle");
  };

  const transcribe = async () => {
    const selectedModel = models.find((model) => model.id === modelId);
    if (!file || !selectedModel) {
      setRequestError(selectedModel ? "invalid_request" : "unknown_model");
      setState("error");
      return;
    }
    if (!selectedModel.supported_languages.includes(language)) {
      setRequestError("incompatible_model_language");
      setState("error");
      return;
    }

    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setResult(null);
    setRequestError(null);
    setState("processing");
    const input: CreateTranscriptionInput = {
      audio: file,
      filename: safeAudioUploadFilename(file),
      language,
      modelId,
      signal: controller.signal,
    };

    try {
      const response = await api.createTranscription(input);
      if (controller.signal.aborted) return;
      if (
        response.status !== "completed" ||
        response.model_id !== modelId ||
        response.language !== language
      ) {
        throw new ApiClientError("unexpected_response", 200, true);
      }
      setResult(response);
      setState("success");
    } catch (error) {
      if (controller.signal.aborted) return;
      setRequestError(errorCode(error));
      setState("error");
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null;
    }
  };

  const modelLabel = result
    ? (models.find((model) => model.id === result.model_id)?.display_name ?? result.model_id)
    : "";
  const controlsDisabled = discovery !== "ready" || state === "processing";
  const canSubmit = Boolean(file && modelId && compatibleModels.length && !controlsDisabled);

  return (
    <div className="page-shell workspace-grid">
      <div>
        <p className="capability-note">{messages.transcribe.integrationNotice}</p>
        <AudioUploader
          disabled={controlsDisabled}
          file={file}
          labels={{
            ...messages.uploader,
            fileTooLarge: messages.errors.fileTooLarge,
            invalidFile: messages.errors.invalidFile,
            or: messages.common.or,
          }}
          onFileChange={(nextFile) => {
            activeRequest.current?.abort();
            setFile(nextFile);
            setResult(null);
            setRequestError(null);
            setState(nextFile ? "ready" : "idle");
          }}
        />
        <div className="panel transcription-controls">
          {discovery === "loading" ? (
            <p aria-live="polite" className="discovery-status" role="status">
              {messages.transcribe.discoveryLoading}
            </p>
          ) : null}
          {discovery === "error" ? (
            <div className="discovery-error" role="alert">
              <p>{messages.transcribe.discoveryError}</p>
              <SecondaryButton
                onClick={() => {
                  setDiscovery("loading");
                  setDiscoveryAttempt((attempt) => attempt + 1);
                }}
              >
                {messages.common.retry}
              </SecondaryButton>
            </div>
          ) : null}
          <LanguageSelector
            disabled={controlsDisabled}
            helpText={messages.transcribe.integrationNotice}
            id="audio-language"
            label={messages.forms.audioLanguage}
            onChange={(event) => {
              const nextLanguage = event.currentTarget.value as LanguageCode;
              const nextModels = models.filter((model) =>
                model.supported_languages.includes(nextLanguage),
              );
              setLanguage(nextLanguage);
              setModelId(nextModels[0]?.id ?? "");
              setResult(null);
              setRequestError(null);
              setState(file ? "ready" : "idle");
            }}
            options={[
              {
                code: "auto",
                disabled: true,
                label: `${messages.common.autoLanguage} — ${messages.status.experimental}`,
              },
              ...languages.map((item) => ({
                code: item.code,
                label: getLanguageName(item.code, locale),
              })),
            ]}
            value={language}
          />
          <div className="field-group">
            <label htmlFor="audio-model">{messages.forms.audioModel}</label>
            <select
              disabled={controlsDisabled || !compatibleModels.length}
              id="audio-model"
              onChange={(event) => {
                setModelId(event.currentTarget.value);
                setResult(null);
                setRequestError(null);
                setState(file ? "ready" : "idle");
              }}
              value={modelId}
            >
              {!compatibleModels.length ? (
                <option value="">{messages.forms.chooseModel}</option>
              ) : null}
              {compatibleModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </div>
          <PrimaryButton
            aria-busy={state === "processing"}
            disabled={!canSubmit}
            onClick={transcribe}
          >
            <Icon name="audio" />
            {state === "processing" ? messages.common.loading : messages.transcribe.action}
          </PrimaryButton>
          <p className="field-help">{messages.uploader.privacyNotice}</p>
        </div>
      </div>
      <section
        aria-busy={state === "processing"}
        aria-labelledby="transcription-result-title"
        className="panel transcription-result-panel"
      >
        <div className="panel-title">
          <Icon name="audio" />
          <h2 id="transcription-result-title">{messages.transcribe.resultTitle}</h2>
        </div>
        {state === "success" && result ? (
          <TranscriptionResult modelLabel={modelLabel} onClear={clear} result={result} />
        ) : null}
        {state === "error" ? (
          <div className="result-error" ref={errorRef} role="alert" tabIndex={-1}>
            <Icon name="x" />
            <h3>{messages.errors.generic}</h3>
            <p>{requestErrorMessage}</p>
            <div className="button-row">
              <PrimaryButton disabled={!file || !modelId} onClick={transcribe}>
                {messages.common.retry}
              </PrimaryButton>
              <SecondaryButton onClick={clear}>{messages.common.clear}</SecondaryButton>
            </div>
          </div>
        ) : null}
        {state === "processing" ? (
          <div aria-live="polite" className="result-placeholder" role="status">
            <span aria-hidden="true" className="loading-spinner" />
            <h3>{messages.transcribe.processingTitle}</h3>
            <p>{messages.transcribe.processingDescription}</p>
          </div>
        ) : null}
        {state === "idle" || state === "ready" ? (
          <div aria-live="polite" className="result-placeholder">
            <Icon name={state === "ready" ? "check" : "headphones"} />
            <h3>
              {state === "ready"
                ? messages.transcribe.readyTitle
                : messages.transcribe.resultEmptyTitle}
            </h3>
            <p>
              {state === "ready"
                ? messages.transcribe.readyDescription
                : messages.transcribe.resultEmptyDescription}
            </p>
          </div>
        ) : null}
      </section>
    </div>
  );
}
