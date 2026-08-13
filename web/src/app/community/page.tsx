"use client";

import { PageHeader } from "@/components/layout/page-header";
import { EmptyState } from "@/components/ui/empty-state";
import { Icon } from "@/components/ui/icon";
import { useI18n } from "@/i18n/provider";

export default function CommunityPage() {
  const { messages } = useI18n();
  const categories = [
    messages.community.learning,
    messages.community.pronunciation,
    messages.community.vocabulary,
    messages.community.culture,
    messages.community.translation,
  ];

  return (
    <>
      <PageHeader
        description={messages.community.description}
        eyebrow={messages.community.eyebrow}
        status="coming_soon"
        statusLabel={messages.status.coming_soon}
        title={messages.community.title}
      />
      <div className="page-shell">
        <EmptyState
          description={`${messages.community.comingSoonNotice} ${messages.community.moderationNotice}`}
          icon="community"
          title={messages.status.coming_soon}
        />
        <section className="content-section" aria-labelledby="community-categories-title">
          <div className="section-heading">
            <div>
              <h2 id="community-categories-title">{messages.community.categoriesTitle}</h2>
            </div>
          </div>
          <div className="info-grid">
            {categories.map((category) => (
              <article className="info-card" key={category}>
                <Icon name="community" />
                <h3>{category}</h3>
                <p>{messages.common.unavailable}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}
