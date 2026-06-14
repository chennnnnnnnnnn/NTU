-- =====================================================================
-- Migration 003: A/B comparison redesign (v3)
-- =====================================================================
-- Replaces the single-clip Likert model with pairwise A/B comparisons.
-- One row per item in a participant's randomised sequence:
--   - 'attention' items (must pick `expected` on the 1–5 scale)
--   - 'comparison' items (play clip_a + clip_b, rate 1–5)
-- Safe to run multiple times (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.survey_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  participant_id uuid NOT NULL REFERENCES public.survey_participants(id) ON DELETE CASCADE,
  order_index int NOT NULL,                 -- position in the randomised sequence (1..N)
  item_type text NOT NULL,                  -- 'attention' | 'comparison'

  -- comparison fields
  sentence_code text,                       -- e.g. "beta1"
  student_id    text,                       -- speaker (voice owner)
  question_type text,                       -- 'q1_accent' | 'q2_human' | 'q3_natural'
  clip_a_id uuid REFERENCES public.survey_clips(id),
  clip_b_id uuid REFERENCES public.survey_clips(id),
  condition_a text,                         -- condition shown as A
  condition_b text,                         -- condition shown as B

  -- attention field
  expected int,                             -- required answer for attention checks

  -- answer
  rating int CHECK (rating BETWEEN 1 AND 5),
  answered_at timestamptz,

  created_at timestamptz NOT NULL DEFAULT now(),

  UNIQUE (participant_id, order_index)
);

CREATE INDEX IF NOT EXISTS idx_survey_items_participant
  ON public.survey_items(participant_id);

ALTER TABLE public.survey_items ENABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE, DELETE ON public.survey_items TO service_role;

-- Result of the attention checks for the whole submission.
ALTER TABLE public.survey_participants
  ADD COLUMN IF NOT EXISTS attention_passed boolean;

-- Refresh PostgREST schema cache so the API sees the new table/columns.
NOTIFY pgrst, 'reload schema';
