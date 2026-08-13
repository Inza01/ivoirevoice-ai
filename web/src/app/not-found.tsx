"use client";

import { EmptyState } from "@/components/ui/empty-state";
import { PrimaryButton } from "@/components/ui/button";
import { useI18n } from "@/i18n/provider";

export default function NotFound() {
  const { messages } = useI18n();

  return (
    <div className="section-shell narrow-page">
      <EmptyState
        action={<PrimaryButton href="/">{messages.common.backHome}</PrimaryButton>}
        description={messages.common.notFoundDescription}
        title={messages.common.notFoundTitle}
      />
    </div>
  );
}
