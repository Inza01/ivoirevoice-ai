"use client";

import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/ui/empty-state";
import { LanguageSelector } from "@/components/ui/language-selector";
import { useI18n } from "@/i18n/provider";
import { getLanguageName } from "@/lib/languages/registry";

export default function ProfilePage() {
  const { locale, messages } = useI18n();

  return (
    <>
      <PageHeader
        description={messages.profile.description}
        eyebrow={messages.profile.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.profile.title}
      />
      <div className="page-shell profile-layout">
        <EmptyState
          description={`${messages.profile.guestDescription} ${messages.profile.authenticationUnavailable}`}
          icon="profile"
          title={messages.profile.guestTitle}
        />
        <aside className="profile-card">
          <div className="profile-avatar" aria-hidden="true">
            IV
          </div>
          <h2>{messages.profile.preferences}</h2>
          <LanguageSelector
            defaultValue={locale}
            disabled
            id="profile-locale"
            label={messages.forms.interfaceLanguage}
            options={[
              { code: "fr", label: getLanguageName("fr", locale) },
              { code: "en", label: getLanguageName("en", locale) },
            ]}
          />
          <ul className="profile-list">
            <li>
              {messages.profile.studiedLanguage} : {getLanguageName("dyu", locale)} (
              {messages.profile.plannedValue})
            </li>
            <li>{messages.profile.level} : —</li>
            <li>{messages.profile.progress} : —</li>
            <li>{messages.profile.recentActivity} : —</li>
          </ul>
        </aside>
      </div>
    </>
  );
}
