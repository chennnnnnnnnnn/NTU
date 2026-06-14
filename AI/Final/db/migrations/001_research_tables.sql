-- =====================================================================
-- Migration 001: research_* tables for Shadow Your Perfect Self study
-- =====================================================================
-- Target: same Supabase project as mirror-app, public schema
-- Convention: research_* prefix to isolate from mirror-app's tables
-- Generated: 2026-05-24, based on codex audit recommendations
-- =====================================================================

-- ─────────────────────────────────────────────────────────────────────
-- 1. research_participants
--    One row per study participant. Anonymous (no email required).
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.research_participants (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Identification
  external_code text UNIQUE NOT NULL,
    -- e.g. "ABC123XY" — given to participant in invitation URL
  user_id uuid REFERENCES auth.users(id) ON DELETE SET NULL,
    -- optional FK if participant has a mirror-app account

  -- Voice IDs from external providers
  fish_voice_id text,
    -- Fish Audio voice_id from mirror-app onboarding
  elevenlabs_voice_id text,
    -- ElevenLabs voice_id created by researcher via IVC

  -- Experimental assignment
  latin_square_group int CHECK (latin_square_group BETWEEN 1 AND 6),

  -- Consent + status
  consent_version text NOT NULL DEFAULT 'v1.0',
  consented_at timestamptz,
  status text NOT NULL DEFAULT 'enrolled',
    -- enrolled / pre_done / train_done / post_done / delayed_done / withdrawn

  -- Demographics (minimal)
  age_bracket text,
  cefr_self_report text,

  -- Audit
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_research_participants_code ON public.research_participants(external_code);
CREATE INDEX idx_research_participants_user ON public.research_participants(user_id);


-- ─────────────────────────────────────────────────────────────────────
-- 2. research_stimuli
--    Master list of the 18 Friends sentences + per-condition audio URLs.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.research_stimuli (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  stimulus_code text UNIQUE NOT NULL,  -- "alpha1", "beta2", "gamma6"
  set_label text NOT NULL CHECK (set_label IN ('alpha', 'beta', 'gamma')),

  sentence text NOT NULL,
  source_episode text,                  -- "S5:E5"
  syllable_count int,
  target_phoneme_count int,
  target_phonemes jsonb,                -- {"theta_eth": 2, "v": 1, ...}

  -- C1 audio is shared across participants (native original or ALT re-recording)
  c1_storage_path text,                 -- e.g. 'research_stimuli/c1_native/alpha1.mp3'

  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_research_stimuli_set ON public.research_stimuli(set_label);


-- ─────────────────────────────────────────────────────────────────────
-- 3. research_assignments
--    Maps each participant to their per-stimulus condition + test stages.
--    Pre-populated when participant enrols (one row per stimulus per test_stage).
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.research_assignments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  participant_id uuid NOT NULL REFERENCES public.research_participants(id) ON DELETE CASCADE,
  stimulus_id uuid NOT NULL REFERENCES public.research_stimuli(id),

  condition int NOT NULL CHECK (condition IN (1, 2, 3)),
    -- 1 = native stranger (C1)
    -- 2 = self + native accent (C2, via ElevenLabs Voice Changer)
    -- 3 = self + Chinese accent (C3, via Fish TTS)

  test_stage text NOT NULL CHECK (test_stage IN ('pre', 'train', 'post', 'delayed')),

  order_index int NOT NULL,
    -- within this test_stage for this participant, what's the presentation order

  -- For C2 / C3, the participant-specific audio URL
  audio_storage_path text,
    -- e.g. 'research_stimuli/c2_eleven_s2s/<participant_id>/alpha1.mp3'

  is_trained boolean NOT NULL DEFAULT FALSE,
    -- TRUE if this stimulus is in participant's training set (12 trained + 6 transfer)

  created_at timestamptz NOT NULL DEFAULT now(),

  UNIQUE (participant_id, stimulus_id, test_stage)
);

CREATE INDEX idx_research_assignments_participant ON public.research_assignments(participant_id);
CREATE INDEX idx_research_assignments_stage ON public.research_assignments(participant_id, test_stage);


-- ─────────────────────────────────────────────────────────────────────
-- 4. research_recordings
--    Each shadow attempt by participant → uploaded to Supabase Storage.
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.research_recordings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  participant_id uuid NOT NULL REFERENCES public.research_participants(id) ON DELETE CASCADE,
  assignment_id uuid REFERENCES public.research_assignments(id) ON DELETE SET NULL,

  -- File location in private `research_recordings/` bucket
  storage_bucket text NOT NULL DEFAULT 'research_recordings',
  storage_path text NOT NULL,
    -- e.g. '<participant_id>/<test_stage>/<stimulus_id>_trial<n>.webm'

  mime_type text,
  duration_sec numeric,
  client_recorded_at timestamptz,

  -- For trials-to-criterion analysis later
  trial_n int,                          -- 1..10 within training; null for pre/post

  -- Post-hoc PPG-DTW results (filled in by analysis pipeline)
  ppg_dtw_distance numeric,
  ppg_dtw_passed_threshold boolean,
  ppg_dtw_computed_at timestamptz,

  uploaded_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_research_recordings_participant ON public.research_recordings(participant_id);
CREATE INDEX idx_research_recordings_assignment ON public.research_recordings(assignment_id);


-- ─────────────────────────────────────────────────────────────────────
-- 5. research_events
--    Generic event log for any session activity (page view, drop, error).
-- ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.research_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  participant_id uuid REFERENCES public.research_participants(id) ON DELETE CASCADE,
    -- nullable for pre-enrolment events (e.g. invalid code attempts)

  event_type text NOT NULL,
    -- consent_signed | page_view | recording_started | recording_ended
    -- upload_success | upload_failed | session_completed | survey_submitted
    -- error | other

  event_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  client_ts timestamptz,
  server_ts timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_research_events_participant ON public.research_events(participant_id);
CREATE INDEX idx_research_events_type ON public.research_events(event_type);


-- =====================================================================
-- RLS Policies
-- =====================================================================
-- Strategy:
--   1. All research_* tables have RLS enabled.
--   2. Anon (browser) access is BLOCKED by default — research-website uses
--      a SHORT-LIVED service-role wrapper at the server-side that scopes to
--      research_* tables only.
--   3. Future option: when research-website also supports authenticated
--      Supabase users (post-pilot), per-participant SELECT policies can be added.

ALTER TABLE public.research_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_stimuli ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_recordings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_events ENABLE ROW LEVEL SECURITY;

-- Default DENY (no explicit policies = nothing accessible to anon/authenticated)
-- Service role bypasses RLS by default in Supabase.

-- (Optional, for future auth integration):
-- CREATE POLICY "participants own row select"
-- ON public.research_participants
-- FOR SELECT
-- TO authenticated
-- USING (auth.uid() = user_id);


-- =====================================================================
-- updated_at trigger for research_participants
-- =====================================================================
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_research_participants_updated_at ON public.research_participants;
CREATE TRIGGER trg_research_participants_updated_at
  BEFORE UPDATE ON public.research_participants
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();


-- =====================================================================
-- Storage buckets (NOTE: must be created via Supabase Dashboard or
-- Storage API; the SQL below is reference only)
-- =====================================================================
-- bucket 'research_stimuli':
--   - public: false
--   - allowed_mime_types: ['audio/mpeg', 'audio/mp4', 'audio/wav']
--   - file_size_limit: 5 MB
--   - signed_url for client read
--
-- bucket 'research_recordings':
--   - public: false
--   - allowed_mime_types: ['audio/webm', 'audio/mp4', 'audio/wav']
--   - file_size_limit: 10 MB
--   - signed_url for client write (write-only signed URL from backend)


-- =====================================================================
-- Account-deletion cascade (TODO when mirror-app /api/account/delete
-- is updated to also delete research_participants where user_id matches)
-- =====================================================================
-- This SQL is handled by FK ON DELETE CASCADE / SET NULL above.
-- Verify mirror-app delete route hits research_participants when user_id matches:
--   mirror-app endpoint: /api/account/delete
--   (file:line ref in codex audit doc)
