import Link from "next/link";

export function Brand({ homeLabel }: { homeLabel: string }) {
  return (
    <Link aria-label={`IvoireVoice — ${homeLabel}`} className="brand" href="/">
      <span className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>IvoireVoice</span>
    </Link>
  );
}
