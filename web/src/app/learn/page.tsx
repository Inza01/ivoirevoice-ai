"use client";

import { PageHeader } from "@/components/layout/page-header";
import { CourseCard } from "@/components/learning/course-card";
import { ProgressCard } from "@/components/learning/progress-card";
import { PrimaryButton } from "@/components/ui/button";
import { Icon } from "@/components/ui/icon";
import { StatusBadge } from "@/components/ui/status-badge";
import { useI18n } from "@/i18n/provider";
import { DEMO_COURSES } from "@/lib/learning/demo-content";

export default function LearnPage() {
  const { locale, messages } = useI18n();

  return (
    <>
      <PageHeader
        actions={
          <PrimaryButton href="/learn/courses">
            {messages.learn.browseCourses}
            <Icon name="arrow-right" />
          </PrimaryButton>
        }
        description={messages.learn.description}
        eyebrow={messages.learn.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.learn.title}
      />
      <div className="page-shell learn-layout">
        <div>
          <div className="demo-banner">{messages.common.demoNeedsReview}</div>
          <section aria-labelledby="recommended-courses-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">{messages.learn.welcomeGuest}</p>
                <h2 id="recommended-courses-title">{messages.learn.recommendedCoursesTitle}</h2>
              </div>
            </div>
            <div className="course-grid">
              {DEMO_COURSES.slice(0, 2).map((course) => (
                <CourseCard
                  description={course.description[locale]}
                  href={`/learn/courses/${course.id}`}
                  key={course.id}
                  level={course.level[locale]}
                  linkLabel={messages.courses.openCourse}
                  progressLabel={messages.common.progressLocal}
                  statusLabel={messages.status.coming_soon}
                  title={course.title[locale]}
                />
              ))}
            </div>
          </section>
        </div>
        <aside className="learning-sidebar" aria-label={messages.learn.progressTitle}>
          <ProgressCard
            detail={messages.learn.signInNotice}
            label={messages.learn.progressTitle}
            value={0}
          />
          <article className="info-card">
            <div className="card-header-row">
              <h3>{messages.learn.levelTitle}</h3>
              <StatusBadge label={messages.status.coming_soon} status="coming_soon" />
            </div>
            <p>{messages.learn.levelDisclaimer}</p>
          </article>
        </aside>
      </div>
    </>
  );
}
