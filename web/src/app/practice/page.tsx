"use client";

import { PageHeader } from "@/components/layout/page-header";
import { Icon, type IconName } from "@/components/ui/icon";
import { StatusBadge } from "@/components/ui/status-badge";
import { useI18n } from "@/i18n/provider";

export default function PracticePage() {
  const { messages } = useI18n();
  const exercises: Array<[string, IconName]> = [
    [messages.practice.multipleChoice, "check"],
    [messages.practice.fillBlank, "book"],
    [messages.practice.matching, "translate"],
    [messages.practice.listeningChoice, "headphones"],
    [messages.practice.listeningTranscription, "audio"],
    [messages.practice.frenchToDioula, "globe"],
    [messages.practice.dioulaToFrench, "globe"],
    [messages.practice.pronunciation, "microphone"],
  ];

  return (
    <>
      <PageHeader
        description={messages.practice.description}
        eyebrow={messages.practice.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.practice.title}
      />
      <div className="page-shell">
        <section aria-labelledby="exercise-types-title">
          <div className="section-heading">
            <div>
              <h2 id="exercise-types-title">{messages.practice.exerciseTypes}</h2>
              <p>{messages.common.demoNeedsReview}</p>
            </div>
          </div>
          <div className="info-grid exercise-card-grid">
            {exercises.map(([label, icon]) => (
              <article className="info-card" key={label}>
                <div className="card-header-row">
                  <span className="feature-icon">
                    <Icon name={icon} />
                  </span>
                  <StatusBadge label={messages.status.coming_soon} status="coming_soon" />
                </div>
                <h3>{label}</h3>
                <p>{messages.common.unavailable}</p>
              </article>
            ))}
          </div>
          <p className="capability-note pronunciation-note">
            {messages.practice.pronunciationDisclaimer}
          </p>
        </section>
      </div>
    </>
  );
}
