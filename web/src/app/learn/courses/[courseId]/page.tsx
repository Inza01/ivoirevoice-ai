"use client";

import { use } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Icon } from "@/components/ui/icon";
import { useI18n } from "@/i18n/provider";
import { getDemoCourse } from "@/lib/learning/demo-content";

export default function CourseDetailPage({ params }: { params: Promise<{ courseId: string }> }) {
  const { courseId } = use(params);
  const { locale, messages } = useI18n();
  const course = getDemoCourse(courseId);

  if (!course) {
    return (
      <div className="section-shell narrow-page">
        <EmptyState
          action={
            <SecondaryButton href="/learn/courses">{messages.learn.browseCourses}</SecondaryButton>
          }
          description={messages.courseDetail.notFoundDescription}
          title={messages.courseDetail.notFoundTitle}
        />
      </div>
    );
  }

  return (
    <>
      <PageHeader
        description={course.description[locale]}
        eyebrow={`${messages.courseDetail.eyebrow} · ${course.level[locale]}`}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={course.title[locale]}
      />
      <div className="page-shell lesson-layout">
        <section className="lesson-card" aria-labelledby="modules-title">
          <div className="demo-banner">{messages.common.demoNeedsReview}</div>
          <h2 id="modules-title">{messages.courseDetail.modules}</h2>
          <ul className="module-list">
            {course.modules.map((moduleTitle, index) => (
              <li key={moduleTitle.fr}>
                <span className="module-index">{index + 1}</span>
                <div>
                  <strong>{moduleTitle[locale]}</strong>
                  <p>{messages.common.structureOnly}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>
        <aside>
          <article className="info-card">
            <h3>{messages.courseDetail.objectives}</h3>
            <p>{messages.learn.previewObjective}</p>
            <PrimaryButton href="/learn/lessons/dire-bonjour">
              <Icon name="book" />
              {messages.courseDetail.start}
            </PrimaryButton>
            <p className="field-help">{messages.courseDetail.progressUnavailable}</p>
          </article>
        </aside>
      </div>
    </>
  );
}
