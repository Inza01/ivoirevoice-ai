"use client";

import { CourseCard } from "@/components/learning/course-card";
import { PageHeader } from "@/components/layout/page-header";
import { LanguageSelector } from "@/components/ui/language-selector";
import { useI18n } from "@/i18n/provider";
import { DEMO_COURSES } from "@/lib/learning/demo-content";

export default function CoursesPage() {
  const { locale, messages } = useI18n();

  return (
    <>
      <PageHeader
        description={messages.courses.description}
        eyebrow={messages.courses.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.courses.title}
      />
      <div className="page-shell">
        <div className="demo-banner">{messages.common.demoNeedsReview}</div>
        <section className="panel course-filters" aria-label={messages.courses.filtersLabel}>
          <LanguageSelector
            defaultValue="all"
            id="course-level"
            label={messages.courses.filterLevel}
            options={[{ code: "all", label: messages.courses.allFilters }]}
          />
          <LanguageSelector
            defaultValue="all"
            id="course-topic"
            label={messages.courses.filterTopic}
            options={[{ code: "all", label: messages.courses.allFilters }]}
          />
        </section>
        <div className="course-grid course-catalog-grid">
          {DEMO_COURSES.map((course) => (
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
      </div>
    </>
  );
}
