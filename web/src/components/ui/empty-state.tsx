import type { ReactNode } from "react";

import { Icon, type IconName } from "@/components/ui/icon";

type EmptyStateProps = {
  action?: ReactNode;
  description: string;
  icon?: IconName;
  title: string;
};

export function EmptyState({ action, description, icon = "spark", title }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon">
        <Icon name={icon} />
      </span>
      <h2>{title}</h2>
      <p>{description}</p>
      {action ? <div className="empty-state-action">{action}</div> : null}
    </div>
  );
}
