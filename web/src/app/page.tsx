"use client";

import { CourseCard } from "@/components/learning/course-card";
import { FeatureCard } from "@/components/marketing/feature-card";
import { Hero } from "@/components/marketing/hero";
import { Icon, type IconName } from "@/components/ui/icon";
import { LanguageBadge } from "@/components/ui/language-badge";
import { StatusBadge } from "@/components/ui/status-badge";
import { useI18n } from "@/i18n/provider";
import { DEMO_COURSES } from "@/lib/learning/demo-content";
import { LANGUAGE_REGISTRY, type CapabilityStatus } from "@/lib/languages/registry";

type SectionHeadingProps = {
  description: string;
  id: string;
  title: string;
};

function SectionHeading({ description, id, title }: SectionHeadingProps) {
  return (
    <header className="section-heading">
      <div>
        <h2 id={id}>{title}</h2>
        <p>{description}</p>
      </div>
    </header>
  );
}

type TechnologyItemProps = {
  description: string;
  icon: IconName;
  status: CapabilityStatus;
  statusLabel: string;
  title: string;
};

function TechnologyItem({ description, icon, status, statusLabel, title }: TechnologyItemProps) {
  return (
    <article className="info-card">
      <div className="card-header-row">
        <span className="feature-icon">
          <Icon name={icon} />
        </span>
        <StatusBadge label={statusLabel} status={status} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
    </article>
  );
}

export default function HomePage() {
  const { locale, messages } = useI18n();

  const features = [
    {
      description: messages.home.transcribeDescription,
      href: "/transcribe",
      icon: "audio",
      status: "experimental",
      title: messages.home.transcribeTitle,
    },
    {
      description: messages.home.translateDescription,
      href: "/translate",
      icon: "translate",
      status: "coming_soon",
      title: messages.home.translateTitle,
    },
    {
      description: messages.home.learnDescription,
      href: "/learn",
      icon: "learn",
      status: "coming_soon",
      title: messages.home.learnTitle,
    },
  ] as const;

  const technologyItems = [
    {
      description: messages.home.transcribeDescription,
      icon: "microphone",
      status: "experimental",
      title: messages.home.transcribeTitle,
    },
    {
      description: messages.home.translateDescription,
      icon: "translate",
      status: "coming_soon",
      title: messages.home.translateTitle,
    },
    {
      description: messages.home.learnDescription,
      icon: "book",
      status: "coming_soon",
      title: messages.home.learnTitle,
    },
  ] as const;

  return (
    <>
      <Hero
        description={messages.home.description}
        eyebrow={messages.home.eyebrow}
        primaryAction={messages.home.primaryAction}
        privacyNote={messages.home.localPrivacyNote}
        secondaryAction={messages.home.secondaryAction}
        title={messages.home.title}
        visualCaption={messages.home.visualCaption}
      />

      <section aria-labelledby="home-features-title" className="home-section">
        <div className="section-shell">
          <SectionHeading
            description={messages.home.featureDescription}
            id="home-features-title"
            title={messages.home.featureTitle}
          />
          <div className="feature-grid">
            {features.map((feature) => (
              <FeatureCard
                description={feature.description}
                href={feature.href}
                icon={feature.icon}
                key={feature.href}
                linkLabel={messages.common.learnMore}
                status={feature.status}
                statusLabel={messages.status[feature.status]}
                title={feature.title}
              />
            ))}
          </div>
        </div>
      </section>

      <section aria-labelledby="home-languages-title" className="home-section">
        <div className="section-shell">
          <SectionHeading
            description={messages.home.languagesDescription}
            id="home-languages-title"
            title={messages.home.languagesTitle}
          />
          <div className="language-grid">
            {Object.values(LANGUAGE_REGISTRY).map((language) => (
              <LanguageBadge
                code={language.code}
                key={language.code}
                label={language.name[locale]}
                status={language.asr}
                statusLabel={messages.status[language.asr]}
              />
            ))}
          </div>
          <p>{messages.home.futureLanguages}</p>
        </div>
      </section>

      <section aria-labelledby="home-courses-title" className="home-section">
        <div className="section-shell">
          <SectionHeading
            description={messages.home.coursesDescription}
            id="home-courses-title"
            title={messages.home.coursesTitle}
          />
          <div className="course-grid">
            {DEMO_COURSES.map((course) => (
              <CourseCard
                description={course.description[locale]}
                href={`/learn/courses/${course.id}`}
                key={course.id}
                level={course.level[locale]}
                linkLabel={messages.courses.openCourse}
                progressLabel={messages.common.progressLocal}
                statusLabel={messages.common.demoNeedsReview}
                title={course.title[locale]}
              />
            ))}
          </div>
        </div>
      </section>

      <section aria-labelledby="home-technology-title" className="home-section">
        <div className="section-shell">
          <SectionHeading
            description={messages.home.technologyDescription}
            id="home-technology-title"
            title={messages.home.technologyTitle}
          />
          <div className="info-grid">
            {technologyItems.map((item) => (
              <TechnologyItem
                description={item.description}
                icon={item.icon}
                key={item.title}
                status={item.status}
                statusLabel={messages.status[item.status]}
                title={item.title}
              />
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
