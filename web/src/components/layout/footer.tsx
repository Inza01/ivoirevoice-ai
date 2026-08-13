"use client";

import Link from "next/link";

import { Brand } from "@/components/layout/brand";
import { useI18n } from "@/i18n/provider";

export function Footer() {
  const { messages } = useI18n();

  return (
    <footer className="site-footer">
      <div className="footer-shell">
        <div className="footer-intro">
          <Brand homeLabel={messages.navigation.home} />
          <p>{messages.footer.description}</p>
          <p className="footer-experimental">{messages.footer.experimental}</p>
        </div>
        <nav aria-label={messages.footer.secondaryNavigation} className="footer-links">
          <div>
            <h2>{messages.footer.explore}</h2>
            <Link href="/transcribe">{messages.navigation.transcribe}</Link>
            <Link href="/translate">{messages.navigation.translate}</Link>
            <Link href="/learn">{messages.navigation.learn}</Link>
          </div>
          <div>
            <h2>IvoireVoice</h2>
            <Link href="/about">{messages.navigation.about}</Link>
            <Link href="/community">{messages.navigation.community}</Link>
            <span>{messages.footer.contactPending}</span>
          </div>
        </nav>
      </div>
      <div className="footer-bottom">
        <p>
          © {new Date().getFullYear()} {messages.footer.localResearch}
        </p>
        <p>{messages.footer.madeInCoteDIvoire}</p>
      </div>
    </footer>
  );
}
