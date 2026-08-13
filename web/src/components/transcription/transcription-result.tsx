"use client";

import { useEffect, useRef, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { useI18n } from "@/i18n/provider";
import type { TranscriptionResponse } from "@/lib/api/contracts";
import { getLanguageName } from "@/lib/languages/registry";
import {
  TRANSCRIPTION_JSON_FILENAME,
  TRANSCRIPTION_TXT_FILENAME,
  downloadText,
  serializeTranscriptionJson,
  serializeTranscriptionTxt,
} from "@/lib/transcription/export";

type TranscriptionResultProps = {
  modelLabel: string;
  onClear: () => void;
  result: TranscriptionResponse;
};

export function TranscriptionResult({ modelLabel, onClear, result }: TranscriptionResultProps) {
  const { locale, messages } = useI18n();
  const resultRef = useRef<HTMLDivElement>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">("idle");
  const text = result.text ?? "";
  const number = new Intl.NumberFormat(locale, { maximumFractionDigits: 2 });

  useEffect(() => {
    resultRef.current?.focus();
  }, [result.id]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
  };

  return (
    <div className="transcription-result" ref={resultRef} tabIndex={-1}>
      <div className="result-success-heading">
        <Icon name="check" />
        <h3>{messages.transcribe.successTitle}</h3>
      </div>
      <p className="result-text">{text || "—"}</p>
      <dl className="result-metadata">
        <div>
          <dt>{messages.forms.audioLanguage}</dt>
          <dd>{getLanguageName(result.language, locale)}</dd>
        </div>
        <div>
          <dt>{messages.forms.audioModel}</dt>
          <dd>{modelLabel}</dd>
        </div>
        <div>
          <dt>{messages.common.duration}</dt>
          <dd>{number.format(result.audio_duration_seconds ?? 0)} s</dd>
        </div>
        <div>
          <dt>{messages.common.processingTime}</dt>
          <dd>{number.format(result.processing_time_seconds ?? 0)} s</dd>
        </div>
        {result.rtf === undefined ? null : (
          <div>
            <dt>RTF</dt>
            <dd>{number.format(result.rtf)}</dd>
          </div>
        )}
      </dl>
      <div className="result-actions">
        <PrimaryButton disabled={!text} onClick={copy}>
          <Icon name="copy" />
          {messages.common.copy}
        </PrimaryButton>
        <SecondaryButton
          disabled={!text}
          onClick={() =>
            downloadText(
              serializeTranscriptionTxt(result),
              TRANSCRIPTION_TXT_FILENAME,
              "text/plain;charset=utf-8",
            )
          }
        >
          <Icon name="download" />
          {messages.transcribe.downloadTxt}
        </SecondaryButton>
        <SecondaryButton
          disabled={!text}
          onClick={() =>
            downloadText(
              serializeTranscriptionJson(result),
              TRANSCRIPTION_JSON_FILENAME,
              "application/json;charset=utf-8",
            )
          }
        >
          <Icon name="download" />
          {messages.transcribe.downloadJson}
        </SecondaryButton>
        <SecondaryButton onClick={onClear}>
          <Icon name="x" />
          {messages.common.clear}
        </SecondaryButton>
      </div>
      <div aria-live="polite" className="result-action-status">
        {copyState === "copied" ? messages.transcribe.copySuccess : null}
      </div>
      {copyState === "error" ? (
        <p className="field-error" role="alert">
          {messages.errors.copyFailed}
        </p>
      ) : null}
      <div className="translation-coming-soon">
        <SecondaryButton disabled>
          <Icon name="translate" />
          {messages.transcribe.translateAction} — {messages.status.coming_soon}
        </SecondaryButton>
      </div>
      <p className="field-help">{messages.transcribe.deleteNotice}</p>
    </div>
  );
}
