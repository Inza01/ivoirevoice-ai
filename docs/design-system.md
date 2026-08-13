# IvoireVoice design system

## Purpose

This document defines the visual and interaction foundation for the new
IvoireVoice web platform. It applies to transcription, translation and learning
journeys while the legacy Gradio interface continues to exist separately.

The system is intentionally calm, warm and restrained. It must remain usable by
people with limited technical experience, on small screens and with assistive
technologies. It targets WCAG 2.2 level AA. Passing automated checks alone is not
sufficient: keyboard, screen-reader, zoom and mobile interaction checks remain
part of each feature's acceptance criteria.

The Foundation axe test runs structural WCAG 2.2 A/AA rules under jsdom. Its
colour-contrast rule is disabled because jsdom has no layout/canvas engine;
contrast evidence comes from the ratios documented below and still requires a
browser-level review before a public release.

## Design principles

1. Make the next action obvious and keep each screen focused on one user goal.
2. Prefer whitespace and hierarchy over additional cards, borders or colours.
3. Use plain language; explain experimental and unavailable capabilities
   explicitly.
4. Never communicate state by colour alone. Pair colour with text and, when
   useful, an icon.
5. Build mobile-first, with progressive enhancement for larger screens.
6. Use native semantic HTML before adding ARIA.
7. Represent African languages with care, without decorative stereotypes or
   unreviewed cultural claims.

## Colour

### Core tokens

| Token | Value | Intended use |
| --- | --- | --- |
| `canvas` | `#F8FAFC` | Page background |
| `surface` | `#FFFFFF` | Cards, dialogs and form surfaces |
| `surface-muted` | `#F1F5F9` | Secondary groups and inactive regions |
| `ink` | `#172033` | Headings and primary body text |
| `ink-muted` | `#475569` | Secondary text and metadata |
| `primary` | `#2563EB` | Primary actions, links and focus indicators |
| `primary-hover` | `#1D4ED8` | Hover state for primary controls |
| `primary-active` | `#1E40AF` | Pressed state and selected text |
| `border-subtle` | `#CBD5E1` | Decorative separators only |
| `border-control` | `#64748B` | Form-control boundaries |
| `success` | `#047857` | Success text, icons and solid actions |
| `warning` | `#B45309` | Warning text and icons |
| `danger` | `#B91C1C` | Error text and destructive actions |
| `accent-decorative` | `#F59E0B` | Small decorative highlights only |

`accent-decorative` has only a 2.15:1 contrast ratio on white. It must not be
used for normal text, an essential icon, a form boundary or the sole indication
of state. Likewise, `border-subtle` is decorative; interactive control
boundaries use `border-control` or another boundary with at least 3:1 contrast.

### Semantic surfaces

| State | Background | Foreground | Contrast |
| --- | --- | --- | ---: |
| Information / selected | `#DBEAFE` | `#1E40AF` | 7.15:1 |
| Success | `#D1FAE5` | `#065F46` | 6.78:1 |
| Warning | `#FEF3C7` | `#854D0E` | 6.15:1 |
| Error | `#FEE2E2` | `#991B1B` | 6.80:1 |
| Neutral | `#F1F5F9` | `#334155` | 9.45:1 |

### Verified text contrasts

Ratios below use the WCAG relative-luminance formula.

| Foreground | Background | Contrast | Result |
| --- | --- | ---: | --- |
| `ink` `#172033` | `surface` `#FFFFFF` | 16.27:1 | AAA |
| `ink` `#172033` | `canvas` `#F8FAFC` | 15.55:1 | AAA |
| `ink-muted` `#475569` | `surface` `#FFFFFF` | 7.58:1 | AAA |
| `ink-muted` `#475569` | `canvas` `#F8FAFC` | 7.24:1 | AAA |
| White `#FFFFFF` | `primary` `#2563EB` | 5.17:1 | AA normal text |
| White `#FFFFFF` | `primary-hover` `#1D4ED8` | 6.70:1 | AA normal text |
| White `#FFFFFF` | `primary-active` `#1E40AF` | 8.72:1 | AAA |
| `success` `#047857` | White `#FFFFFF` | 5.48:1 | AA normal text |
| `warning` `#B45309` | White `#FFFFFF` | 5.02:1 | AA normal text |
| `danger` `#B91C1C` | White `#FFFFFF` | 6.47:1 | AA normal text |

New colour combinations must be checked before use. Normal text requires at
least 4.5:1; large text requires at least 3:1. Essential graphical objects,
focus rings and interactive boundaries require at least 3:1 against adjacent
colours.

## Typography

Use a system font stack to avoid a blocking remote font request and to retain
good rendering across operating systems:

```css
font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
```

Do not use uppercase for sentences or Dioula examples. Preserve Unicode text in
NFC and test every chosen font with the project's French, English and Dioula
character repertoire.

| Role | Mobile | Desktop | Weight | Line height |
| --- | ---: | ---: | ---: | ---: |
| Display | 36 px | 52 px | 700 | 1.10 |
| Page title | 30 px | 40 px | 700 | 1.15 |
| Section title | 24 px | 30 px | 700 | 1.25 |
| Card title | 18 px | 20 px | 600 | 1.35 |
| Body | 16 px | 16 px | 400 | 1.60 |
| Body strong | 16 px | 16 px | 600 | 1.60 |
| Small / metadata | 14 px | 14 px | 400 | 1.50 |

Body text must not be smaller than 16 px by default. Reserve 14 px for short,
non-essential metadata. Long-form lesson content should remain between 60 and
75 characters per line.

## Spacing and layout

Use a 4 px base unit and this constrained scale:

| Token | Value | Typical use |
| --- | ---: | --- |
| `space-1` | 4 px | Tight icon/text adjustment |
| `space-2` | 8 px | Related inline elements |
| `space-3` | 12 px | Compact control groups |
| `space-4` | 16 px | Default component padding |
| `space-6` | 24 px | Card padding and form groups |
| `space-8` | 32 px | Related content sections |
| `space-12` | 48 px | Page sections on mobile |
| `space-16` | 64 px | Page sections on desktop |
| `space-24` | 96 px | Hero spacing on wide screens |

The content container is at most 1200 px wide, centered, with 16 px horizontal
padding on mobile, 24 px on tablet and 32 px on desktop. Do not fill empty space
with low-value widgets.

## Radius, borders and elevation

| Token | Value | Use |
| --- | ---: | --- |
| `radius-sm` | 8 px | Inputs, compact buttons and badges |
| `radius-md` | 12 px | Default buttons and cards |
| `radius-lg` | 16 px | Feature panels and dialogs |
| `radius-pill` | 9999 px | Status badges only |

Use 1 px borders. `border-subtle` separates non-interactive regions;
`border-control` identifies interactive fields. Avoid fully rounded cards.

```css
--shadow-sm: 0 1px 2px rgb(15 23 42 / 0.06);
--shadow-md: 0 8px 24px rgb(15 23 42 / 0.10);
--shadow-dialog: 0 20px 48px rgb(15 23 42 / 0.16);
```

Cards use `shadow-sm` or no shadow. Reserve `shadow-md` for floating navigation
or a raised interactive panel and `shadow-dialog` for modal dialogs. Elevation
must communicate layering, not decoration.

## Buttons

All buttons use an explicit `<button>` element unless navigation is the actual
action, in which case use a link. The visible label must describe the result,
for example “Transcrire l’audio”, not “Continuer”.

- Minimum visual height: 44 px; minimum touch target: 44 × 44 px.
- Default horizontal padding: 20 px; gap between icon and text: 8 px.
- Primary: `primary` background with white text.
- Secondary: white background, `ink` text and `border-control` border.
- Tertiary: text/link treatment for low-emphasis actions.
- Destructive: `danger` background with white text and a confirmation when the
  action is difficult to undo.
- Icon-only buttons require an accessible name and a visible tooltip on hover
  and keyboard focus.
- A loading button retains its label and dimensions, exposes `aria-busy=true`
  and prevents duplicate submission.
- Disabled controls remain legible and are never the only explanation for why
  an action is unavailable. Show adjacent guidance.

Hover, pressed, focus, loading and disabled states are mandatory. Hover must not
be the sole way to discover an action.

## Cards

Cards group one coherent subject. Do not place every section in a card.

- Default padding is 24 px and radius is 12 px.
- Use one heading level appropriate to the page hierarchy.
- A clickable card must have one clear interactive target, keyboard focus and a
  descriptive name. Avoid nested competing links.
- Course cards expose title, level, short description, progress when known and
  one primary action.
- Empty cards are replaced by an `EmptyState` with explanation and next action.

## Forms

- Every control has a persistent visible `<label>`; placeholder text is never a
  label.
- Required fields are identified in text before submission.
- Instructions and errors are connected with `aria-describedby`.
- Inputs are at least 44 px high, use 16 px text and a visible
  `border-control` boundary.
- Error state combines a message, icon and colour. Preserve the user's input.
- On failed submission, move focus to an error summary, then provide links to
  invalid fields.
- File upload states distinguish idle, drag-over, validating, ready, processing,
  completed and failed. Validate extension, MIME type and size server-side too.
- Microphone permission denial must provide a retry path and file-upload
  alternative.

Never persist user audio by default. Consent and retention choices must be
explicit and separate from the primary submit action.

## Navigation

The primary destinations are Accueil, Transcrire, Traduire, Apprendre,
S’exercer, Communauté and Mon espace. “À propos” may remain in the footer or
secondary navigation.

- Use a semantic `<nav>` with an accessible name.
- Identify the current page with `aria-current="page"`, visible text weight and
  a non-colour cue.
- Desktop navigation stays concise; no mega-menu in the Foundation MVP.
- Mobile navigation uses a labelled menu button, traps no focus when closed,
  closes with Escape and restores focus to its trigger.
- Place a “Aller au contenu” skip link as the first focusable element.
- Keep header actions reachable without horizontal scrolling at 320 px width.

## Status and capability badges

Use exactly three capability states across transcription, translation,
learning and community features:

- `Disponible` / `Available`: success treatment.
- `Expérimental` / `Experimental`: warning treatment and a nearby limitation.
- `Bientôt disponible` / `Coming soon`: neutral treatment; not interactive.

A badge is supplemental context, not the only capability signal. Do not use a
success badge for unvalidated Dioula translation or pronunciation scoring.

## Feedback and system states

- Inline feedback is preferred when it belongs to one field or result.
- Use a polite `aria-live="polite"` region for completed background actions.
- Use `role="alert"` sparingly for failures requiring immediate attention.
- Toasts never contain the only copy of important information and remain long
  enough to read.
- Loading states use concise text plus a progress indicator. Do not invent a
  percentage when progress is indeterminate.
- Error messages explain what happened, what remains safe and what the user can
  do next. Do not expose stack traces, private paths or model internals.
- Offline and unavailable-model states preserve all unaffected actions.

## Focus, keyboard and motion

Every interactive element must be reachable and operable with a keyboard in a
logical DOM order. Do not use positive `tabindex` values.

The standard focus indicator is a 3 px `primary` outline with a 2 px offset. It
must remain visible on light, selected and dark control backgrounds; add a white
separation ring when needed. Never remove outlines without an equivalent or
stronger replacement.

Respect `prefers-reduced-motion: reduce`. With reduced motion enabled, disable
parallax, decorative movement and non-essential transitions. Otherwise, keep
small interface transitions between 120 and 200 ms and never make animation a
prerequisite for understanding state.

Audio controls must have keyboard-operable play, pause and seek functions,
visible time information, and a text alternative for pedagogical content.

## Responsive behaviour

Use mobile-first styles and content-driven layouts. Reference breakpoints are:

| Name | Minimum width | Behaviour |
| --- | ---: | --- |
| Mobile | 0 px | One column, compact header, full-width primary actions |
| Small | 640 px | Wider forms and optional two-column card groups |
| Tablet | 768 px | Persistent navigation where space permits |
| Desktop | 1024 px | Two-panel translation and richer course grids |
| Wide | 1280 px | Constrained 1200 px content container; no stretched text |

- At 320 CSS pixels wide, content must reflow without two-dimensional scrolling,
  except intrinsically two-dimensional content.
- At 200% browser zoom, navigation and form actions remain available.
- Translation panels stack source before target below 1024 px.
- Course grids use one, two and at most three columns as space permits.
- Do not hide core actions on mobile. Microphone and upload controls retain
  44 × 44 px targets with sufficient spacing to avoid accidental activation.

## Discreet African-inspired motif

The identity may use a small original geometric rhythm inspired broadly by
weaving, connection and sound. It is an accent, not proof of cultural
authenticity.

- Use simple original lines, dots or interlocking shapes at no more than 6%
  opacity on large empty surfaces.
- Limit the motif to the hero, a section divider or footer; never repeat it
  behind body text, forms or transcription results.
- Mark decorative artwork `aria-hidden="true"` and remove it from the reading
  order.
- Do not copy a named textile, ethnic symbol, sacred pattern, commercial work or
  another product's visual identity.
- Avoid maps, wildlife silhouettes and generic “tribal” imagery as shortcuts for
  African identity.
- Any culturally specific motif, name or meaning requires documented review by
  relevant cultural and linguistic contributors before publication.
- The interface must remain complete and understandable when the motif is
  removed or when forced-colour mode suppresses it.

## Accessibility definition of done

A page is ready only when all applicable checks below pass:

- semantic landmarks and one clear page-level heading;
- meaningful link and button names;
- full keyboard journey, visible focus and correct focus restoration;
- text and non-text contrast thresholds documented above;
- no information conveyed by colour, position, sound or motion alone;
- labels, instructions, errors and status announcements associated correctly;
- reflow at 320 px and usability at 200% zoom;
- minimum 44 × 44 px product touch targets, exceeding the WCAG 2.2 AA minimum;
- reduced-motion and forced-colour behaviour checked;
- page title and language metadata set correctly;
- French and English interface copy reviewed, with Dioula content marked as
  placeholder until linguistic validation;
- automated accessibility tests supplemented by manual keyboard and
  screen-reader smoke tests.

## Implementation guidance

Expose these values as CSS custom properties and map Tailwind utilities to the
same semantic names. Components consume semantic tokens such as `primary` and
`danger`, never raw palette indices. Dark mode is outside the Foundation MVP;
do not ship an incomplete theme toggle.

Component variants must remain small and explicit. Build accessible primitives
locally before introducing a component generator or runtime styling library.
No design-system dependency may load fonts, analytics, images or scripts from a
third-party service at runtime without a separate security and privacy review.
