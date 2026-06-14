// Server-only: produce + persist research_assignments rows for a participant.
//
// Idempotent: if a (participant, test_stage) tuple already has rows in
// research_assignments, return them as-is. Otherwise compute them via the
// Latin square in ./latin-square.ts and insert before returning.

import { researchServiceClient } from "./supabase/research";
import {
  buildAssignments,
  isLatinSquareGroup,
  pickGroupForCode,
  type LatinSquareGroup,
  type SetLabel,
} from "./latin-square";
import type { Condition } from "./types";

export type ManifestRow = {
  assignment_id: string;
  stimulus_id: string;
  stimulus_code: string;
  set_label: SetLabel;
  sentence: string;
  syllable_count: number | null;
  target_phoneme_count: number | null;
  target_phonemes: Record<string, number> | null;
  condition: Condition;
  test_stage: "pre" | "train" | "post";
  order_index: number;
  is_trained: boolean;
  audio_storage_path: string | null;
};

type ParticipantRow = {
  id: string;
  external_code: string;
  latin_square_group: number | null;
};

type StimulusRow = {
  id: string;
  stimulus_code: string;
  set_label: SetLabel;
  sentence: string;
  syllable_count: number | null;
  target_phoneme_count: number | null;
  target_phonemes: Record<string, number> | null;
};

/**
 * Get the manifest for (participant, stage). Materialises assignments if needed.
 */
export async function getOrCreateManifest(
  participantId: string,
  stage: "pre" | "train" | "post",
): Promise<{ manifest: ManifestRow[]; participant: ParticipantRow }> {
  const supa = researchServiceClient();

  // 1. Load participant; auto-assign group if missing.
  const { data: participant, error: pErr } = await supa
    .from("research_participants")
    .select("id, external_code, latin_square_group")
    .eq("id", participantId)
    .maybeSingle();
  if (pErr) throw new Error(`load participant: ${pErr.message}`);
  if (!participant) throw new Error(`participant not found: ${participantId}`);

  let group: LatinSquareGroup;
  if (isLatinSquareGroup(participant.latin_square_group)) {
    group = participant.latin_square_group as LatinSquareGroup;
  } else {
    group = pickGroupForCode(participant.external_code as string);
    const { error: uErr } = await supa
      .from("research_participants")
      .update({ latin_square_group: group })
      .eq("id", participantId);
    if (uErr) throw new Error(`set group: ${uErr.message}`);
    participant.latin_square_group = group; // reflect newly assigned value in response
  }

  // 2. Check if assignments already exist for this stage.
  const { data: existing, error: aErr } = await supa
    .from("research_assignments")
    .select("id, stimulus_id, condition, test_stage, order_index, is_trained, audio_storage_path")
    .eq("participant_id", participantId)
    .eq("test_stage", stage)
    .order("order_index", { ascending: true });
  if (aErr) throw new Error(`load assignments: ${aErr.message}`);

  // 3. Load all 18 stimuli (needed in either branch).
  const { data: stimuli, error: sErr } = await supa
    .from("research_stimuli")
    .select("id, stimulus_code, set_label, sentence, syllable_count, target_phoneme_count, target_phonemes");
  if (sErr) throw new Error(`load stimuli: ${sErr.message}`);
  if (!stimuli || stimuli.length === 0) {
    throw new Error("research_stimuli is empty; run scripts/seed-stimuli.mjs first");
  }
  const stimulusByCode = new Map<string, StimulusRow>();
  const stimulusById = new Map<string, StimulusRow>();
  for (const s of stimuli as StimulusRow[]) {
    stimulusByCode.set(s.stimulus_code, s);
    stimulusById.set(s.id, s);
  }

  // 4. If we already have assignments for this stage, return them.
  if (existing && existing.length > 0) {
    return {
      participant: participant as ParticipantRow,
      manifest: existing.map((row) => {
        const stim = stimulusById.get(row.stimulus_id as string);
        if (!stim) throw new Error(`stimulus missing for assignment ${row.id}`);
        return rowToManifest(row, stim);
      }),
    };
  }

  // 5. Build assignments for ALL three stages at once (transactional-ish: insert
  //    once, then return only the requested stage). This is the only place
  //    assignments are created for a participant; subsequent stages will hit
  //    branch (4) and just read.
  const shapes = buildAssignments(
    group,
    (stimuli as StimulusRow[]).map((s) => ({
      stimulus_code: s.stimulus_code,
      set_label: s.set_label,
    })),
  );

  const insertRows = shapes.map((sh) => {
    const stim = stimulusByCode.get(sh.stimulus_code);
    if (!stim) throw new Error(`stimulus missing for code ${sh.stimulus_code}`);
    return {
      participant_id: participantId,
      stimulus_id: stim.id,
      condition: sh.condition,
      test_stage: sh.test_stage,
      order_index: sh.order_index,
      is_trained: sh.is_trained,
      audio_storage_path: null,
    };
  });

  const { data: inserted, error: iErr } = await supa
    .from("research_assignments")
    .insert(insertRows)
    .select("id, stimulus_id, condition, test_stage, order_index, is_trained, audio_storage_path");
  if (iErr) throw new Error(`insert assignments: ${iErr.message}`);

  const wanted = (inserted ?? []).filter((r) => r.test_stage === stage);
  wanted.sort((a, b) => (a.order_index as number) - (b.order_index as number));

  return {
    participant: participant as ParticipantRow,
    manifest: wanted.map((row) => {
      const stim = stimulusById.get(row.stimulus_id as string);
      if (!stim) throw new Error(`stimulus missing post-insert: ${row.stimulus_id}`);
      return rowToManifest(row, stim);
    }),
  };
}

type AssignmentDBRow = {
  id: string;
  stimulus_id: string;
  condition: number;
  test_stage: string;
  order_index: number;
  is_trained: boolean;
  audio_storage_path: string | null;
};

function rowToManifest(row: AssignmentDBRow, stim: StimulusRow): ManifestRow {
  return {
    assignment_id: row.id,
    stimulus_id: stim.id,
    stimulus_code: stim.stimulus_code,
    set_label: stim.set_label,
    sentence: stim.sentence,
    syllable_count: stim.syllable_count,
    target_phoneme_count: stim.target_phoneme_count,
    target_phonemes: stim.target_phonemes,
    condition: row.condition as Condition,
    test_stage: row.test_stage as "pre" | "train" | "post",
    order_index: row.order_index,
    is_trained: row.is_trained,
    audio_storage_path: row.audio_storage_path,
  };
}
