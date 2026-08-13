"use client";

import { use } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { PrimaryButton, SecondaryButton } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { Icon } from "@/components/ui/icon";
import { useI18n } from "@/i18n/provider";
import { getDemoLesson } from "@/lib/learning/demo-content";

export default function LessonPage({ params }: { params: Promise<{ lessonId: string }> }) {
  const { lessonId } = use(params);
  const { locale, messages } = useI18n();
  const lesson = getDemoLesson(lessonId);

  if (!lesson) {
    return (
      <div className="section-shell narrow-page">
        <EmptyState
          action={
            <SecondaryButton href="/learn/courses">{messages.learn.browseCourses}</SecondaryButton>
          }
          description={messages.lesson.notFoundDescription}
          title={messages.lesson.notFoundTitle}
        />
      </div>
    );
  }

  return (
    <>
      <PageHeader
        description={lesson.objective[locale]}
        eyebrow={messages.lesson.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={lesson.title[locale]}
      />
      <div className="page-shell lesson-layout">
        <article className="lesson-card lesson-body">
          <div className="demo-banner">{messages.common.demoNeedsReview}</div>
          <h2>{messages.lesson.objective}</h2>
          <p>{lesson.objective[locale]}</p>
          <h2>{messages.lesson.content}</h2>
          <p>{messages.common.lessonStructureDescription}</p>
          <div className="result-placeholder">
            <Icon name="headphones" />
            <h3>{messages.lesson.dioulaExample}</h3>
            <p>{messages.common.demoNeedsReview}</p>
            <SecondaryButton disabled>
              <Icon name="audio" />
              {messages.lesson.listen}
            </SecondaryButton>
          </div>
          <h2>{messages.lesson.quickExercise}</h2>
          <p>{messages.practice.pronunciationDisclaimer}</p>
        </article>
        <aside className="info-card">
          <h3>{messages.lesson.vocabulary}</h3>
          <p>{messages.common.demoNeedsReview}</p>
          <PrimaryButton disabled>{messages.lesson.markComplete}</PrimaryButton>
          <p className="field-help">{messages.lesson.completionUnavailable}</p>
          <div className="lesson-navigation">
            <SecondaryButton href="/learn/courses/premiers-pas">
              {messages.common.previous}
            </SecondaryButton>
            <SecondaryButton href="/learn/lessons/demander-comment-ca-va">
              {messages.common.next}
            </SecondaryButton>
          </div>
        </aside>
      </div>
    </>
  );
}
