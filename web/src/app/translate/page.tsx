"use client";

import { PageHeader } from "@/components/layout/page-header";
import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { LanguageSelector } from "@/components/ui/language-selector";
import { useI18n } from "@/i18n/provider";
import { LANGUAGE_CODES, getLanguageName } from "@/lib/languages/registry";

export default function TranslatePage() {
  const { locale, messages } = useI18n();
  const options = LANGUAGE_CODES.map((code) => ({
    code,
    label: getLanguageName(code, locale),
  }));

  return (
    <>
      <PageHeader
        description={messages.translate.description}
        eyebrow={messages.translate.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.translate.title}
      />
      <div className="page-shell">
        <p className="capability-note">{messages.translate.comingSoonNotice}</p>
        <div className="translate-grid">
          <section className="panel text-panel" aria-labelledby="source-text-title">
            <LanguageSelector
              defaultValue="fr"
              id="source-language"
              label={messages.forms.sourceLanguage}
              options={options}
            />
            <label className="visually-hidden" htmlFor="source-text" id="source-text-title">
              {messages.forms.textToTranslate}
            </label>
            <textarea id="source-text" placeholder={messages.translate.sourcePlaceholder} />
            <SecondaryButton disabled>
              <Icon name="audio" />
              {messages.translate.useTranscription}
            </SecondaryButton>
          </section>
          <div className="swap-control">
            <SecondaryButton aria-label={messages.translate.swapLanguages} disabled>
              <Icon name="translate" />
            </SecondaryButton>
          </div>
          <section className="panel text-panel" aria-labelledby="target-text-title">
            <LanguageSelector
              defaultValue="dyu"
              id="target-language"
              label={messages.forms.targetLanguage}
              options={options}
            />
            <label className="visually-hidden" htmlFor="target-text" id="target-text-title">
              {messages.forms.translationResult}
            </label>
            <textarea
              disabled
              id="target-text"
              placeholder={messages.translate.targetPlaceholder}
            />
            <PrimaryButton disabled>
              <Icon name="translate" />
              {messages.translate.action}
            </PrimaryButton>
          </section>
        </div>
        <p className="field-help translation-provider-note">{messages.translate.providerNotice}</p>
      </div>
    </>
  );
}
