import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import { SiteShell } from "@/components/layout/site-shell";
import { I18nProvider } from "@/i18n/provider";
import { MESSAGES } from "@/i18n/messages";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: `${MESSAGES.fr.brand.name} — ${MESSAGES.fr.brand.tagline}`,
    template: `%s — ${MESSAGES.fr.brand.name}`,
  },
  description: MESSAGES.fr.home.description,
};

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#f8fafc",
  width: "device-width",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="fr">
      <body>
        <I18nProvider>
          <SiteShell>{children}</SiteShell>
        </I18nProvider>
      </body>
    </html>
  );
}
