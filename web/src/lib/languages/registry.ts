export const LANGUAGE_CODES = ["fr", "en", "dyu"] as const;
export type LanguageCode = (typeof LANGUAGE_CODES)[number];

export const UI_LOCALES = ["fr", "en"] as const;
export type UiLocale = (typeof UI_LOCALES)[number];

export const CAPABILITY_STATUSES = ["available", "experimental", "coming_soon"] as const;
export type CapabilityStatus = (typeof CAPABILITY_STATUSES)[number];

export type LanguageCapability = "interface" | "asr" | "learning";

export interface LanguageDefinition {
  readonly code: LanguageCode;
  readonly name: Readonly<Record<UiLocale, string>>;
  readonly interface: CapabilityStatus;
  readonly asr: CapabilityStatus;
  readonly learning: CapabilityStatus;
  readonly translationTargets: Readonly<Partial<Record<LanguageCode, CapabilityStatus>>>;
}

/**
 * Product capabilities exposed by the Phase 2 web platform.
 *
 * ASR is a real local workflow, but remains experimental: French and English
 * have no product benchmark and Dioula evidence is limited to the documented
 * corpus. Runtime model availability always comes from the discovery API.
 */
export const LANGUAGE_REGISTRY: Readonly<Record<LanguageCode, LanguageDefinition>> = {
  fr: {
    code: "fr",
    name: { fr: "Français", en: "French" },
    interface: "available",
    asr: "experimental",
    learning: "coming_soon",
    translationTargets: {
      en: "coming_soon",
      dyu: "coming_soon",
    },
  },
  en: {
    code: "en",
    name: { fr: "Anglais", en: "English" },
    interface: "available",
    asr: "experimental",
    learning: "coming_soon",
    translationTargets: {
      fr: "coming_soon",
      dyu: "coming_soon",
    },
  },
  dyu: {
    code: "dyu",
    name: { fr: "Dioula", en: "Dioula" },
    interface: "coming_soon",
    asr: "experimental",
    learning: "coming_soon",
    translationTargets: {
      fr: "coming_soon",
      en: "coming_soon",
    },
  },
};

export function isLanguageCode(value: string): value is LanguageCode {
  return LANGUAGE_CODES.some((code) => code === value);
}

export function isUiLocale(value: string): value is UiLocale {
  return UI_LOCALES.some((locale) => locale === value);
}

export function getLanguage(code: LanguageCode): LanguageDefinition {
  return LANGUAGE_REGISTRY[code];
}

export function getLanguageName(code: LanguageCode, locale: UiLocale): string {
  return LANGUAGE_REGISTRY[code].name[locale];
}

export function getTranslationStatus(
  source: LanguageCode,
  target: LanguageCode,
): CapabilityStatus | null {
  if (source === target) {
    return null;
  }
  return LANGUAGE_REGISTRY[source].translationTargets[target] ?? null;
}
