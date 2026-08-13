import { describe, expect, it } from "vitest";

import {
  LANGUAGE_CODES,
  LANGUAGE_REGISTRY,
  UI_LOCALES,
  getLanguageName,
  getTranslationStatus,
  isLanguageCode,
  isUiLocale,
} from "@/lib/languages/registry";

describe("language registry", () => {
  it("uses the canonical Phase 2 language and locale codes", () => {
    expect(LANGUAGE_CODES).toEqual(["fr", "en", "dyu"]);
    expect(UI_LOCALES).toEqual(["fr", "en"]);
    expect(isLanguageCode("dyu")).toBe(true);
    expect(isLanguageCode("dy")).toBe(false);
    expect(isUiLocale("dyu")).toBe(false);
  });

  it("localizes language names without changing language identity", () => {
    expect(getLanguageName("fr", "en")).toBe("French");
    expect(getLanguageName("en", "fr")).toBe("Anglais");
    expect(getLanguageName("dyu", "fr")).toBe("Dioula");
  });

  it("labels the real local ASR workflow as experimental", () => {
    for (const language of Object.values(LANGUAGE_REGISTRY)) {
      expect(language.asr).toBe("experimental");
    }
  });

  it("does not simulate or advertise any translation pair", () => {
    for (const source of LANGUAGE_CODES) {
      for (const target of LANGUAGE_CODES) {
        const status = getTranslationStatus(source, target);
        expect(status).toBe(source === target ? null : "coming_soon");
      }
    }
  });
});
