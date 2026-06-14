# research-website — Design Doc (v2, 2026-05-25)

**Project**: Shadow Your Perfect Self (MHCI CSIE7641 Final Project + TAICHI 2026 Poster)
**Build target**: 5/25 ~ 5/31 (7 days)
**Run target**: 6/1 ~ 6/8 (N=20 user study)
**Submission targets**: Part 3 due 6/9; TAICHI poster due 6/18 (EN, sigconf)

---

## 1. Architecture (locked 2026-05-25)

**Pure web-only**, zero mirror-app modification.

```
research-website (this repo)            mirror-app (sister, READ-ONLY, unmodified)
├── calibration (5×60s Chinese)         ├── on App Store (iOS)
├── voice cloning (Fish + ElevenLabs)   ├── exists but not touched
├── pre-test                            └── 0 files modified by this project
├── train (12 stimuli)
├── post-test
└── survey + done
```

Rationale (decided after exploring hybrid + in-app-OTA + web-only):
- TAICHI poster + international submission needs a portable URL artifact
- mirror-app is solo product code, separate from research artifacts
- Single repo simplifies open-source story for paper appendix
- Browser MediaRecorder is sufficient quality for Fish IVC + ElevenLabs IVC

Mirror-app sister path (NEVER MODIFY):
`~/Documents/NTU/NTUCS_AI/2026_Summer_Programs/2026NTUAI_Builders_Challenge/instabrain/mirror-app/`

---

## 2. Locked decisions

| # | Decision |
|---|---|
| D1 | **Web-only.** No cross-app dependency. mirror-app `git status` must stay clean. |
| D2 | **Anonymous + external_code.** One invitation URL `?code=ABC123`; no email/password auth. |
| D3 | **Pre-generated stimuli.** Local batch script produces 738 mp3, uploaded once to `research_stimuli/` bucket. Browser fetches signed URLs only. |
| D4 | **Client-direct upload.** Recordings go to `research_recordings/` via signed upload URL. Backend does not relay audio bytes. |
| D5 | **Calibration in web.** 5×~60s Chinese MediaRecorder, server calls Fish IVC + ElevenLabs IVC to mint both voice_ids. Replaces previous D5 (mirror-app onboarding). |
| D6 | **Same Supabase project as mirror-app, isolated via `research_*` prefix.** `researchServiceClient()` wrapper enforces table whitelist at runtime. |
| D7 | **Academic warm palette + Lora/Plex fonts.** No emoji, no gradients, no SaaS clichés. Rubric in `.review/`. |

---

## 3. User flow (受試者視角)

```
Researcher sends invitation: https://shadow-research.vercel.app/?code=ABC123

  ↓
/  (Landing + consent + invitation code form)
  ↓ POST /api/enroll
/calibration  (5 × Chinese sentences, each ~60s recording)
  ↓ POST /api/calibration-upload (×5)
  ↓ POST /api/finalize-voice
/processing  (poll until both voice_ids ready)
  ↓
/pre-test       ← Phase D (not Phase B)
/train          ← Phase D
/post-test      ← Phase D
/done           ← Phase D
```

Phase B builds: `/`, `/calibration`, `/processing` + 3 APIs + 2 components.

---

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Framework | Next.js **16.2.6** App Router | searchParams is `Promise<...>` (await required) |
| React | 19.2.4 | Server Components default |
| Styling | Tailwind **4.3.0** | `@theme inline` CSS variable mapping |
| Fonts | Lora (serif h1/h2), IBM Plex Sans (body), IBM Plex Mono (counters) | via `next/font/google` |
| Audio | MediaRecorder API + `<audio>` | no third-party lib |
| State | URL searchParams + Server Components | no Redux/Zustand |
| DB | Supabase Postgres (`research_*` tables) | RLS deny-by-default |
| Storage | Supabase Storage | `research_stimuli` public 5 MB, `research_recordings` private 10 MB |
| Auth | External invitation code | no Supabase auth |
| Voice clone | Fish Audio IVC + ElevenLabs IVC | server-side, never browser-exposed |
| Deploy | Vercel | `vercel deploy` from CLI |

---

## 5. Palette + typography

```css
/* defined in src/app/globals.css */
--bg-base:        #FAF7F2  /* warm off-white parchment */
--bg-card:        #F2EDE4  /* subtle card differentiation */
--text-ink:       #1C1814  /* warm near-black */
--text-muted:     #6B635A  /* warm gray */
--border-warm:    #DCD3C5  /* hairline divider */
--accent:         #8B3A2F  /* terracotta, max 2× per screen */

font-serif:  Lora            (h1, h2, page titles)
font-sans:   IBM Plex Sans   (body, buttons)
font-mono:   IBM Plex Mono   (trial counter, time, code)
```

Constraints (anti-AI-slop):
- No gradient, no glassmorphism, no Lottie, no emoji-in-UI
- No centered hero, no "Get Started" / "Welcome to" CTA copy
- Content starts top-left (academic paper feel)
- Single accent (terracotta) ≤2 uses per screen

---

## 6. File structure (Phase B target)

```
src/
├── app/
│   ├── layout.tsx              ← updated: Lora + Plex Sans + Plex Mono
│   ├── globals.css             ← updated: academic warm palette
│   ├── page.tsx                ← rewritten: consent + invitation code form
│   ├── calibration/
│   │   └── page.tsx            ← Phase B: 5-segment recording
│   ├── processing/
│   │   └── page.tsx            ← Phase B: voice-clone polling
│   └── api/
│       ├── enroll/route.ts                  ← Phase B
│       ├── calibration-upload/route.ts       ← Phase B
│       └── finalize-voice/route.ts           ← Phase B (Fish + ElevenLabs IVC)
├── components/
│   ├── AudioRecorder.tsx       ← Phase B reusable component
│   └── SegmentPrompt.tsx       ← Phase B
└── lib/
    ├── supabase/
    │   ├── client.ts           (existing)
    │   ├── server.ts           (existing)
    │   └── research.ts         (existing, table whitelist)
    ├── types.ts                (existing)
    └── calibration-script.ts   ← Phase B: 5 Chinese segments + phoneme targets
```

---

## 7. API surface (Phase B)

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/enroll` | POST | `{ code, consent_version }` | `{ participant_id }` |
| `/api/calibration-upload` | POST | multipart: `{ participant_id, segment_n, blob }` | `{ storage_path }` |
| `/api/finalize-voice` | POST | `{ participant_id }` | `{ fish_voice_id, elevenlabs_voice_id }` or `{ status: 'processing' }` |

All routes use `researchServiceClient()` (service role, table-whitelisted).
All routes validate input. No secrets returned to browser.

---

## 8. Calibration script (5 segments)

Each segment is a Mandarin sentence designed to:
- Cover both 4 tones and key Mandarin phoneme contrasts
- Be ~30-50 syllables (giving 30-60s of speech for IVC)
- Naturally include English phonemes that L1 Mandarin speakers will mis-render in their L2 (informs C3 generation)

Defined in `src/lib/calibration-script.ts`.

---

## 9. Phase plan (Phase B = current)

| Phase | Scope | Hours | Status |
|---|---|---|---|
| **A** | Scaffold + migration + GitHub push | 6 | ✅ done |
| **B** | Consent + calibration + voice cloning + processing | 8 | 🟡 in progress |
| C | Pre-test (cold-read 6 stimuli) + upload | 6 | ⏳ |
| D | Train + post + done + survey | 8 | ⏳ |
| E | Pre-generate stimuli + Vercel deploy + pilot | 4 | ⏳ |

After Phase B, STOP for human review per Rubric in `.review/phase-B-final.md`.

---

## 10. Rubric (4 dimensions, target ≥4/5 each)

Full 42-criterion rubric documented in goal prompt v0.2 (this session).
Summary:
- **A. Design language consistency** — single palette, single font system, single button shape
- **B. Originality** — no SaaS clichés, no AI design slop, no decorative gradients
- **C. Technical execution** — h-12 buttons, focus ring, ARIA labels, no CLS
- **D. Research-grade usability** — progress indicator, trial counter, explicit errors, no marketing language

---

## 11. References (existing, unchanged)

- Codex audit (mirror-app): `~/Documents/NTU/NTUCS_AI/HCI LAB/office hour/2026-05-24_codex-audit_comprehensive-build-prep.md`
- Part 2 paper bundle: `final_project/part2/overleaf_upload.zip`
- Stimuli (Phase D needs): `final_project/stimuli/stimuli_18_friends.md`
- Figure 1 (paper): `final_project/figures/Figure1_pipeline.svg`
- TAICHI 2026 CFP: https://taichi2026.taiwanchi.org/cfp/
