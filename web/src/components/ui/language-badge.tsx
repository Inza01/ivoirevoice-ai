import { StatusBadge } from "@/components/ui/status-badge";
import type { CapabilityStatus, LanguageCode } from "@/lib/languages/registry";

type LanguageBadgeProps = {
  code: LanguageCode;
  label: string;
  status: CapabilityStatus;
  statusLabel: string;
};

export function LanguageBadge({ code, label, status, statusLabel }: LanguageBadgeProps) {
  return (
    <article className="language-badge">
      <span className="language-monogram" aria-hidden="true">
        {code.slice(0, 2).toUpperCase()}
      </span>
      <span className="language-copy">
        <strong>{label}</strong>
        <span>{code}</span>
      </span>
      <StatusBadge label={statusLabel} status={status} />
    </article>
  );
}
