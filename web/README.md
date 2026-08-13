# IvoireVoice web platform

This directory contains the isolated Next.js frontend for the Phase 2 language platform. Its
transcription page communicates with IvoireVoice through public FastAPI contracts; it never imports
Python model code or reads corpus, checkpoint or prediction directories.

## Runtime

- Node.js `24.19.0` (LTS), pinned in `.nvmrc` and `package.json`;
- Next.js `16.3.0` with the App Router;
- React `19.2.8`;
- TypeScript in strict mode;
- Tailwind CSS `4.1.18` through PostCSS;
- ESLint, Prettier and Vitest with jsdom.

All npm dependency versions are exact and the reviewed `package-lock.json` provides a deterministic
install. Dependency directories and build outputs are ignored; use `npm ci` after the initial
reviewed resolution.

## Install and verify

```bash
cd web
nvm use
npm ci
npm run verify
```

CI and subsequent local installations use `npm ci`, not an unlocked dependency resolution.

## Commands

```bash
npm run dev           # local development server
npm run build         # production build with the stable webpack path
npm run start         # serve a completed production build
npm run lint          # ESLint, with zero warnings allowed
npm run typecheck     # strict TypeScript check without output
npm run test          # deterministic Vitest suite
npm run audit         # reject high or critical dependency advisories
npm run format:check  # formatting check without rewriting files
npm run verify        # gates, online dependency audit, then production build
```

Both local servers bind to `127.0.0.1` by default. Exposing the development or production server on
another interface must be an explicit deployment decision with an appropriate reverse proxy.

Next.js 16 does not run ESLint as part of `next build`; `npm run verify` keeps lint and build as
separate mandatory gates. The root `make web-verify` is the offline developer gate. CI runs
`npm run audit` separately with registry access before executing the same build checks. The Next
build uses webpack and the in-process TypeScript API so it remains usable in restricted workers that
prohibit detached child processes; it still performs Next's complete type gate.

## Configuration

Copy `.env.example` to a local ignored environment file and set `IVOIREVOICE_API_INTERNAL_URL` to
the private FastAPI HTTP(S) origin. It must be an origin without credentials, a path, a query or a
fragment. This value is server-only. Browser code calls the same-origin route handler at
`/api/backend`; it never receives private hosts, filesystem paths or credentials through
`NEXT_PUBLIC_*` variables.

The route handler only proxies the Foundation contracts for health, model and language discovery,
transcription creation/status and translation creation. It forwards neither cookies nor browser
authorization headers. Mutating calls require a same-origin `Origin` header, request sizes are
bounded (25 MiB of audio plus a small multipart allowance), upstream calls time out, and public
error bodies contain stable codes only. The handler does not log or persist request or response
content.

The ASR page accepts WAV, MP3, FLAC and OGG up to 25 MiB; FastAPI performs the authoritative decode
and 30-second duration validation. It uses runtime discovery for `fr`, `en`, `dyu` and compatible
models. Auto-detection and microphone capture remain disabled. Copy and TXT/JSON downloads happen
only in the browser and exclude the source filename and private model metadata.

Run FastAPI and Next.js from the repository root in two terminals:

```bash
make api
make web
```

No user audio is retained by this frontend or backend. Translation, learning content and
pronunciation scoring remain unavailable or explicitly `coming_soon` until their providers and
content have been validated. This local MVP has no authentication or rate limiting and must not be
exposed directly to the Internet.
