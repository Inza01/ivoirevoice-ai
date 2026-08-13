import Link from "next/link";

import { Icon, type IconName } from "@/components/ui/icon";
import { StatusBadge } from "@/components/ui/status-badge";
import type { CapabilityStatus } from "@/lib/languages/registry";

type FeatureCardProps = {
  description: string;
  href: string;
  icon: IconName;
  linkLabel: string;
  status: CapabilityStatus;
  statusLabel: string;
  title: string;
};

export function FeatureCard({
  description,
  href,
  icon,
  linkLabel,
  status,
  statusLabel,
  title,
}: FeatureCardProps) {
  return (
    <article className="feature-card">
      <div className="card-header-row">
        <span className="feature-icon">
          <Icon name={icon} />
        </span>
        <StatusBadge label={statusLabel} status={status} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <Link className="text-link" href={href}>
        {linkLabel}
        <Icon name="arrow-right" />
      </Link>
    </article>
  );
}
