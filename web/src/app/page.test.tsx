import { render, screen } from "@testing-library/react";

import HomePage from "@/app/page";
import { I18nProvider } from "@/i18n/provider";
import { MESSAGES } from "@/i18n/messages";

describe("HomePage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("presents the foundation without claiming unavailable capabilities", () => {
    const messages = MESSAGES.fr;

    render(
      <I18nProvider initialLocale="fr">
        <HomePage />
      </I18nProvider>,
    );

    expect(
      screen.getByRole("heading", { level: 1, name: messages.home.title }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: messages.home.primaryAction })).toHaveAttribute(
      "href",
      "/transcribe",
    );
    expect(screen.getAllByText(messages.common.demoNeedsReview)).toHaveLength(3);
    expect(screen.getAllByText(messages.status.experimental)).toHaveLength(5);
    expect(screen.getAllByText(messages.status.coming_soon)).toHaveLength(4);
  });

  it("uses the central registry to expose all three languages", () => {
    render(
      <I18nProvider initialLocale="en">
        <HomePage />
      </I18nProvider>,
    );

    expect(screen.getByText("French")).toBeInTheDocument();
    expect(screen.getByText("English")).toBeInTheDocument();
    expect(screen.getByText("Dioula")).toBeInTheDocument();
  });

  it("links every demo card to an existing public course", () => {
    render(
      <I18nProvider initialLocale="fr">
        <HomePage />
      </I18nProvider>,
    );

    expect(
      screen.getAllByRole("link", { name: MESSAGES.fr.courses.openCourse })[0],
    ).toHaveAttribute("href", "/learn/courses/premiers-pas");
  });
});
