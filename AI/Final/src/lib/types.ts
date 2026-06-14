// Database types for research_* tables (mirrors db/migrations/001_research_tables.sql).
//
// TODO (Phase B): regenerate via `supabase gen types typescript` once migration
// has run in the live Supabase project. For now, manual mirror.

export type ResearchParticipant = {
  id: string                     // uuid
  external_code: string          // invitation code (URL ?code=ABC123)
  user_id: string | null
  fish_voice_id: string | null
  elevenlabs_voice_id: string | null
  latin_square_group: 1 | 2 | 3 | 4 | 5 | 6 | null
  consent_version: string
  consented_at: string | null    // ISO timestamp
  status: 'enrolled' | 'pre_done' | 'train_done' | 'post_done' | 'delayed_done' | 'withdrawn'
  age_bracket: string | null
  cefr_self_report: string | null
  created_at: string
  updated_at: string
}

export type ResearchStimulus = {
  id: string
  stimulus_code: string          // "alpha1", "beta2"
  set_label: 'alpha' | 'beta' | 'gamma'
  sentence: string
  source_episode: string | null
  syllable_count: number | null
  target_phoneme_count: number | null
  target_phonemes: Record<string, number> | null
  c1_storage_path: string | null
  created_at: string
}

export type Condition = 1 | 2 | 3
export type TestStage = 'pre' | 'train' | 'post' | 'delayed'

export type ResearchAssignment = {
  id: string
  participant_id: string
  stimulus_id: string
  condition: Condition
  test_stage: TestStage
  order_index: number
  audio_storage_path: string | null
  is_trained: boolean
  created_at: string
}

export type ResearchRecording = {
  id: string
  participant_id: string
  assignment_id: string | null
  storage_bucket: string
  storage_path: string
  mime_type: string | null
  duration_sec: number | null
  client_recorded_at: string | null
  trial_n: number | null
  ppg_dtw_distance: number | null
  ppg_dtw_passed_threshold: boolean | null
  ppg_dtw_computed_at: string | null
  uploaded_at: string
}

export type ResearchEvent = {
  id: string
  participant_id: string | null
  event_type:
    | 'consent_signed'
    | 'page_view'
    | 'recording_started'
    | 'recording_ended'
    | 'upload_success'
    | 'upload_failed'
    | 'session_completed'
    | 'survey_submitted'
    | 'error'
    | 'other'
  event_payload: Record<string, unknown>
  client_ts: string | null
  server_ts: string
}
