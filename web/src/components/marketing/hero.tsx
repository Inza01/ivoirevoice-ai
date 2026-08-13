import { Icon } from "@/components/ui/icon";
import { PrimaryButton, SecondaryButton } from "@/components/ui/button";

type HeroProps = {
  description: string;
  eyebrow: string;
  primaryAction: string;
  privacyNote: string;
  secondaryAction: string;
  title: string;
  visualCaption: string;
};

export function Hero({
  description,
  eyebrow,
  primaryAction,
  privacyNote,
  secondaryAction,
  title,
  visualCaption,
}: HeroProps) {
  return (
    <section className="hero section-shell" aria-labelledby="hero-title">
      <div className="hero-copy">
        <p className="eyebrow">
          <Icon name="spark" />
          {eyebrow}
        </p>
        <h1 id="hero-title">{title}</h1>
        <p className="hero-description">{description}</p>
        <div className="button-row">
          <PrimaryButton href="/transcribe">
            {primaryAction}
            <Icon name="arrow-right" />
          </PrimaryButton>
          <SecondaryButton href="/learn">{secondaryAction}</SecondaryButton>
        </div>
        <p className="hero-note">
          <span aria-hidden="true">●</span>
          {privacyNote}
        </p>
      </div>
      <div className="hero-visual" aria-hidden="true">
        <div className="visual-orbit visual-orbit-one" />
        <div className="visual-orbit visual-orbit-two" />
        <div className="sound-card">
          <span className="sound-card-label">IVOIREVOICE</span>
          <Icon className="sound-card-icon" name="audio" />
          <div className="sound-bars">
            {[16, 28, 42, 24, 52, 36, 20, 44, 31, 18, 38, 26].map((height, index) => (
              <span key={`${height}-${index}`} style={{ height }} />
            ))}
          </div>
          <p>{visualCaption}</p>
        </div>
        <div className="pattern-tile pattern-tile-one" />
        <div className="pattern-tile pattern-tile-two" />
      </div>
    </section>
  );
}
