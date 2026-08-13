"use client";

import { PageHeader } from "@/components/layout/page-header";
import { TranscriptionExperience } from "@/components/transcription/transcription-experience";
import { useI18n } from "@/i18n/provider";

export default function TranscribePage() {
  const { messages } = useI18n();

  return (
    <>
      <PageHeader
        description={messages.transcribe.description}
        eyebrow={messages.transcribe.eyebrow}
        status="experimental"
        statusLabel={messages.status.experimental}
        title={messages.transcribe.title}
      />
      <TranscriptionExperience />
    </>
  );
}
