import { describe, expect, it } from "vitest";

import { MESSAGES, getMessages } from "@/i18n/messages";

function leafKeys(value: object, prefix = ""): string[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof child === "object" && child !== null ? leafKeys(child, path) : [path];
  });
}

describe("message catalogs", () => {
  it("keeps French and English catalogs structurally aligned", () => {
    expect(leafKeys(MESSAGES.fr)).toEqual(leafKeys(MESSAGES.en));
  });

  it("provides accessible navigation labels in both locales", () => {
    expect(getMessages("fr").navigation.skipToContent).toBeTruthy();
    expect(getMessages("en").navigation.openMenu).toBeTruthy();
    expect(getMessages("en").navigation.closeMenu).toBeTruthy();
    expect(getMessages("fr").common.or).toBe("ou");
    expect(getMessages("en").common.or).toBe("or");
  });

  it("externalizes copy for every Foundation route and shared form", () => {
    const sections = [
      "footer",
      "uploader",
      "forms",
      "home",
      "transcribe",
      "translate",
      "learn",
      "courses",
      "courseDetail",
      "lesson",
      "practice",
      "community",
      "profile",
      "about",
    ] as const;

    for (const section of sections) {
      expect(Object.keys(getMessages("fr")[section]).length).toBeGreaterThan(0);
      expect(Object.keys(getMessages("en")[section]).length).toBeGreaterThan(0);
    }
  });

  it("labels unreviewed learning material explicitly", () => {
    expect(getMessages("fr").common.demoNeedsReview).toContain("À VALIDER");
    expect(getMessages("en").common.demoNeedsReview).toContain("TO BE REVIEWED");
  });

  it("does not claim that coming-soon capabilities are available", () => {
    expect(getMessages("fr").status.coming_soon).not.toBe(getMessages("fr").status.available);
    expect(getMessages("en").status.coming_soon).toBe("Coming soon");
    expect(getMessages("fr").transcribe.integrationNotice).toContain("pas encore");
    expect(getMessages("en").translate.comingSoonNotice).toContain("No translation engine");
  });

  it("keeps learning content and pronunciation limitations explicit", () => {
    expect(getMessages("fr").courses.demoStatus).toContain("démonstration");
    expect(getMessages("en").courses.demoStatus).toContain("Demonstration");
    expect(getMessages("fr").practice.pronunciationDisclaimer).toContain("Aucun score phonétique");
    expect(getMessages("en").learn.levelDisclaimer).toContain("do not represent CEFR");
    expect(getMessages("fr").common.progressLocal).toBe("Progression locale");
    expect(getMessages("en").profile.plannedValue).toBe("planned");
  });
});
