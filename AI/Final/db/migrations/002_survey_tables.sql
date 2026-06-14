-- =====================================================================
-- Migration 002: survey_* tables for the audio-rating survey
-- =====================================================================
-- A standalone, lightweight survey that reuses the same Supabase project
-- and the same `*_prefix isolation strategy as research_* (migration 001).
--
-- Flow: enroll → profile (demographics) → rate 10 random clips → done.
-- Independent of the shadow-shadowing study; does NOT touch research_* or
-- mirror-app tables.
-- =====================================================================

-- ─────────────────────────────────────────────────────────────────────
-- 1. survey_participants — one row per respondent (anonymous + demographics)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.survey_participants (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  external_code text UNIQUE,            -- invitation code, or auto-generated anon code
  status text NOT NULL DEFAULT 'enrolled',
    -- enrolled / profile_done / in_progress / completed

  -- Start-page fields
  team_member_name text,                -- required name (v3)
  student_id    text,                   -- (legacy) voice owner
  stimulus_set  text,                   -- (legacy) alpha / beta / gamma
  person_num    text,                   -- "1".."4"

  -- Demographics (collected on the profile page; all optional)
  age_bracket     text,                 -- "18-24" / "25-34" / "35-44" / "45+"
  gender          text,                 -- "male" / "female" / "prefer_not"
  native_language text,
  english_level   text,                 -- CEFR self-report A1..C2
  used_headphones boolean,              -- answering with headphones?

  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_survey_participants_code
  ON public.survey_participants(external_code);


-- ─────────────────────────────────────────────────────────────────────
-- 2. survey_clips — the pool of audio files (18 native-TTS clips)
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.survey_clips (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  clip_code    text UNIQUE NOT NULL,    -- "<student>_<set><n>_<cond>" e.g. r14922a21_beta1_c2
  sentence     text,                    -- the line spoken in the clip
  storage_path text NOT NULL,           -- "/clips/<student>/<set><n>_<cond>.wav"

  -- Per-clip classification (for batch selection + analysis)
  student_id    text,                   -- voice owner
  stimulus_set  text,                   -- alpha / beta / gamma
  sentence_code text,                   -- "beta1"
  condition     text,                   -- c2 / c3a / c3b

  created_at timestamptz NOT NULL DEFAULT now()
);


-- ─────────────────────────────────────────────────────────────────────
-- 3. survey_responses — one row per (participant, assigned clip)
--    Pre-populated (answered_at = null) when the participant finishes the
--    profile page; UPDATEd in place as they rate each clip.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.survey_responses (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  participant_id uuid NOT NULL REFERENCES public.survey_participants(id) ON DELETE CASCADE,
  clip_id        uuid NOT NULL REFERENCES public.survey_clips(id),
  order_index    int  NOT NULL,         -- 1..10, fixed at assignment time

  -- Likert ratings (1..5); null until answered
  rating_naturalness int CHECK (rating_naturalness BETWEEN 1 AND 5),
  rating_clarity     int CHECK (rating_clarity     BETWEEN 1 AND 5),
  rating_nativeness  int CHECK (rating_nativeness  BETWEEN 1 AND 5),
  rating_overall     int CHECK (rating_overall     BETWEEN 1 AND 5),

  listen_count int DEFAULT 0,           -- how many times they pressed play
  answered_at  timestamptz,             -- null = assigned but not yet answered

  created_at timestamptz NOT NULL DEFAULT now(),

  UNIQUE (participant_id, clip_id)
);

CREATE INDEX IF NOT EXISTS idx_survey_responses_participant
  ON public.survey_responses(participant_id);


-- =====================================================================
-- RLS — deny-by-default (same strategy as research_* tables).
-- Browser/anon access is blocked; all access goes through the server-side
-- service-role wrapper (src/lib/supabase/survey.ts).
-- =====================================================================
ALTER TABLE public.survey_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.survey_clips        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.survey_responses    ENABLE ROW LEVEL SECURITY;

-- service_role bypasses RLS but still needs table-level GRANTs. Some fresh
-- Supabase projects do not auto-grant on SQL-Editor-created tables, so make
-- it explicit. (anon/authenticated stay denied — no grants = no access.)
GRANT SELECT, INSERT, UPDATE, DELETE
  ON public.survey_participants, public.survey_clips, public.survey_responses
  TO service_role;


-- =====================================================================
-- updated_at trigger for survey_participants
-- Self-contained: defines set_updated_at() here so this migration can run
-- in a fresh Supabase project without depending on migration 001.
-- =====================================================================
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_survey_participants_updated_at ON public.survey_participants;
CREATE TRIGGER trg_survey_participants_updated_at
  BEFORE UPDATE ON public.survey_participants
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();


-- =====================================================================
-- Reload the PostgREST schema cache so the API immediately sees any
-- newly-added columns (otherwise reads/writes fail with PGRST204/204).
-- =====================================================================
NOTIFY pgrst, 'reload schema';
