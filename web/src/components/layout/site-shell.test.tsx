import { render } from "@testing-library/react";
import axe from "axe-core";
import { vi } from "vitest";

import { SiteShell } from "@/components/layout/site-shell";
import { I18nProvider } from "@/i18n/provider";

vi.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

describe("SiteShell", () => {
  beforeEach(() => window.localStorage.clear());

  it("has no automatically detectable structural WCAG 2.2 AA violations", async () => {
    const { container } = render(
      <I18nProvider initialLocale="fr">
        <SiteShell>
          <h1>Page de contrôle</h1>
          <p>Contenu synthétique.</p>
        </SiteShell>
      </I18nProvider>,
    );

    const results = await axe.run(container, {
      rules: {
        "color-contrast": { enabled: false },
      },
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"] },
    });

    expect(results.violations).toEqual([]);
  });
});
