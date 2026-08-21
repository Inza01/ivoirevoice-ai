import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import TranscribePage from "@/app/transcribe/page";
import { I18nProvider } from "@/i18n/provider";
import { MESSAGES } from "@/i18n/messages";

vi.mock("@/components/transcription/transcription-experience", () => ({
  TranscriptionExperience: () => <div data-testid="transcription-experience" />,
}));

describe("TranscribePage", () => {
  it("presents the real local workflow as experimental", () => {
    render(
      <I18nProvider initialLocale="fr">
        <TranscribePage />
      </I18nProvider>,
    );

    expect(
      screen.getByRole("heading", { level: 1, name: MESSAGES.fr.transcribe.title }),
    ).toBeInTheDocument();
    expect(screen.getByText(MESSAGES.fr.status.experimental)).toBeInTheDocument();
    expect(screen.queryByText(MESSAGES.fr.status.coming_soon)).not.toBeInTheDocument();
    expect(screen.getByTestId("transcription-experience")).toBeInTheDocument();
  });
});
