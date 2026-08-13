"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { MESSAGES, type MessageCatalog } from "@/i18n/messages";
import type { UiLocale } from "@/lib/languages/registry";

type I18nContextValue = {
  locale: UiLocale;
  messages: MessageCatalog;
  setLocale: (locale: UiLocale) => void;
};

const I18nContext = createContext<I18nContextValue | null>(null);

type I18nProviderProps = {
  children: ReactNode;
  initialLocale?: UiLocale;
};

export function I18nProvider({ children, initialLocale = "fr" }: I18nProviderProps) {
  const [locale, setLocale] = useState<UiLocale>(initialLocale);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const value = useMemo(() => ({ locale, messages: MESSAGES[locale], setLocale }), [locale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within I18nProvider");
  }
  return context;
}
