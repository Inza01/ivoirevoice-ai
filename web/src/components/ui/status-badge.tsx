import type { CapabilityStatus } from "@/lib/languages/registry";

type StatusBadgeProps = {
  label: string;
  status: CapabilityStatus;
};

export function StatusBadge({ label, status }: StatusBadgeProps) {
  return <span className={`status-badge status-${status}`}>{label}</span>;
}
