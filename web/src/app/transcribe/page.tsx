"use client";

import { PageHeader } from "@/components/layout/page-header";
import { AudioUploader } from "@/components/transcription/audio-uploader";
import { PrimaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { LanguageSelector } from "@/components/ui/language-selector";
import { useI18n } from "@/i18n/provider";
import { LANGUAGE_CODES, getLanguageName } from "@/lib/languages/registry";

export default function TranscribePage() {
  const { locale, messages } = useI18n();
  const options = [
    { code: "auto", label: messages.common.autoLanguage, disabled: true },
    ...LANGUAGE_CODES.map((code) => ({
      code,
      label: getLanguageName(code, locale),
    })),
  ];

  return (
    <>
      <PageHeader
        description={messages.transcribe.description}
        eyebrow={messages.transcribe.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.transcribe.title}
      />
      <div className="page-shell workspace-grid">
        <div>
          <p className="capability-note">{messages.transcribe.integrationNotice}</p>
          <AudioUploader
            labels={{
              ...messages.uploader,
              fileTooLarge: messages.errors.fileTooLarge,
              invalidFile: messages.errors.invalidFile,
              or: messages.common.or,
            }}
          />
          <div className="panel transcription-controls">
            <LanguageSelector
              defaultValue="dyu"
              helpText={messages.transcribe.integrationNotice}
              id="audio-language"
              label={messages.forms.audioLanguage}
              options={options}
            />
            <PrimaryButton disabled>
              <Icon name="audio" />
              {messages.transcribe.action}
            </PrimaryButton>
            <p className="field-help">{messages.uploader.privacyNotice}</p>
          </div>
        </div>
        <section className="panel" aria-labelledby="transcription-result-title">
          <div className="panel-title">
            <Icon name="audio" />
            <h2 id="transcription-result-title">{messages.transcribe.resultTitle}</h2>
          </div>
          <div className="result-placeholder" aria-live="polite">
            <Icon name="headphones" />
            <h3>{messages.transcribe.resultEmptyTitle}</h3>
            <p>{messages.transcribe.resultEmptyDescription}</p>
          </div>
        </section>
      </div>
    </>
  );
}
