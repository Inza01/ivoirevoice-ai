"use client";

import { PageHeader } from "@/components/layout/page-header";
import { Icon, type IconName } from "@/components/ui/icon";
import { StatusBadge } from "@/components/ui/status-badge";
import { useI18n } from "@/i18n/provider";

export default function AboutPage() {
  const { locale, messages } = useI18n();
  const percent = new Intl.NumberFormat(locale, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
    style: "percent",
  });
  const decimal = new Intl.NumberFormat(locale, {
    maximumFractionDigits: 5,
    minimumFractionDigits: 5,
  });
  const integer = new Intl.NumberFormat(locale);
  const sections: Array<[string, string, IconName]> = [
    [messages.about.missionTitle, messages.about.missionDescription, "spark"],
    [messages.about.privacyTitle, messages.about.privacyDescription, "check"],
    [messages.about.limitationsTitle, messages.about.limitationsDescription, "audio"],
    [messages.about.futureTitle, messages.about.futureDescription, "globe"],
  ];

  return (
    <>
      <PageHeader
        description={messages.about.description}
        eyebrow={messages.about.eyebrow}
        title={messages.about.title}
      />
      <div className="page-shell">
        <section className="info-grid" aria-label={messages.about.capabilitiesTitle}>
          {sections.map(([title, description, icon]) => (
            <article className="info-card" key={title}>
              <span className="feature-icon">
                <Icon name={icon} />
              </span>
              <h3>{title}</h3>
              <p>{description}</p>
            </article>
          ))}
        </section>
        <section className="content-section" aria-labelledby="public-results-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">{messages.about.frozenModelLabel}</p>
              <h2 id="public-results-title">{messages.about.resultsTitle}</h2>
              <p>{messages.about.resultsDescription}</p>
            </div>
            <StatusBadge label={messages.status.experimental} status="experimental" />
          </div>
          <div className="stat-grid">
            <div className="stat-card">
              <strong>{percent.format(0.3326)}</strong>
              <span>{messages.about.werLabel}</span>
            </div>
            <div className="stat-card">
              <strong>{percent.format(0.1238)}</strong>
              <span>{messages.about.cerLabel}</span>
            </div>
            <div className="stat-card">
              <strong>{decimal.format(0.00785)}</strong>
              <span>{messages.about.rtfLabel}</span>
            </div>
            <div className="stat-card">
              <strong>{integer.format(2624)}</strong>
              <span>{messages.about.holdoutLabel}</span>
            </div>
          </div>
        </section>
      </div>
    </>
  );
}
