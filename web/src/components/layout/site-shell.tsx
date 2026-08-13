"use client";

import type { ReactNode } from "react";

import { Footer } from "@/components/layout/footer";
import { Navbar } from "@/components/layout/navbar";
import { useI18n } from "@/i18n/provider";

export function SiteShell({ children }: { children: ReactNode }) {
  const { messages } = useI18n();

  return (
    <>
      <a className="skip-link" href="#main-content">
        {messages.navigation.skipToContent}
      </a>
      <Navbar />
      <main id="main-content">{children}</main>
      <Footer />
    </>
  );
}
