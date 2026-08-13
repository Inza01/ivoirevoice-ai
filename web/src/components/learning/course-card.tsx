import Link from "next/link";

import { Icon } from "@/components/ui/icon";
import { StatusBadge } from "@/components/ui/status-badge";

type CourseCardProps = {
  description: string;
  href: string;
  level: string;
  linkLabel: string;
  progress?: number;
  progressLabel: string;
  statusLabel: string;
  title: string;
};

export function CourseCard({
  description,
  href,
  level,
  linkLabel,
  progress,
  progressLabel,
  statusLabel,
  title,
}: CourseCardProps) {
  return (
    <article className="course-card">
      <div className="course-art" aria-hidden="true">
        <Icon name="learn" />
        <span>{level}</span>
      </div>
      <div className="course-content">
        <div className="card-header-row">
          <span className="course-level">{level}</span>
          <StatusBadge label={statusLabel} status="coming_soon" />
        </div>
        <h3>{title}</h3>
        <p>{description}</p>
        {typeof progress === "number" ? (
          <div className="compact-progress">
            <div className="progress-copy">
              <span>{progressLabel}</span>
              <strong>{progress}%</strong>
            </div>
            <progress max="100" value={progress}>
              {progress}%
            </progress>
          </div>
        ) : null}
        <Link className="text-link" href={href}>
          {linkLabel}
          <Icon name="arrow-right" />
        </Link>
      </div>
    </article>
  );
}
