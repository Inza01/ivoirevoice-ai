import type { ReactNode } from "react";

import { StatusBadge } from "@/components/ui/status-badge";
import type { CapabilityStatus } from "@/lib/languages/registry";

type PageHeaderProps = {
  actions?: ReactNode;
  description: string;
  eyebrow: string;
  status?: CapabilityStatus;
  statusLabel?: string;
  title: string;
};

export function PageHeader({
  actions,
  description,
  eyebrow,
  status,
  statusLabel,
  title,
}: PageHeaderProps) {
  return (
    <header className="page-header section-shell">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <div className="page-title-row">
          <h1>{title}</h1>
          {status && statusLabel ? <StatusBadge label={statusLabel} status={status} /> : null}
        </div>
        <p>{description}</p>
      </div>
      {actions ? <div className="page-header-actions">{actions}</div> : null}
    </header>
  );
}
