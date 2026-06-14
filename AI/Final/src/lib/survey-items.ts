// Server-only: build + persist a participant's randomised item sequence
// (2 attention checks + 18 A/B comparisons), and read it back.
//
// Sampling rules (see survey-config):
//   - 2 sentences from each set α/β/γ (pool excludes gamma5, beta6) = 6
//   - each sentence assigned one speaker; no speaker used more than 2×
//   - each sentence → 3 comparisons (q1 c2/c3b, q2 c3a/c3b, q3 c2/c3a)
//   - A/B order per item randomised; the 18 comparisons shuffled; the 2
//     attention checks placed first.
//
// Idempotent: if items already exist for the participant, return them.

import { surveyServiceClient } from "./supabase/survey";
import {
  SETS,
  SENTENCES_PER_SET,
  MAX_SPEAKER_USES,
  SPEAKERS,
  COMPARISON_QUESTIONS,
  ATTENTION_CHECKS,
  sentencePool,
} from "./survey-config";

export type SurveyItem = {
  item_id: string;
  order_index: number;
  item_type: "attention" | "comparison";
  answered: boolean;
  // comparison only:
  question_type?: string;
  sentence?: string | null;
  clip_a_url?: string;
  clip_b_url?: string;
  // attention only:
  prompt?: string;
};

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

type ClipRow = {
  id: string;
  student_id: string;
  sentence_code: string;
  condition: string;
  sentence: string | null;
  storage_path: string;
};

export async function getOrCreateSurveyItems(
  participantId: string,
): Promise<SurveyItem[]> {
  const supa = surveyServiceClient();

  // 1. Resume if already built.
  const { data: existing, error: eErr } = await supa
    .from("survey_items")
    .select(
      "id, order_index, item_type, answered_at, question_type, expected, sentence_code, clip_a:survey_clips!clip_a_id(storage_path, sentence), clip_b:survey_clips!clip_b_id(storage_path)",
    )
    .eq("participant_id", participantId)
    .order("order_index", { ascending: true });
  if (eErr) throw new Error(`load items: ${eErr.message}`);
  if (existing && existing.length > 0) return existing.map(rowToItem);

  // 2. Sample 2 sentences per set.
  const sentenceCodes: string[] = [];
  for (const set of SETS) {
    sentenceCodes.push(...shuffle([...sentencePool(set)]).slice(0, SENTENCES_PER_SET));
  }

  // 3. Assign a speaker to each sentence, capped at MAX_SPEAKER_USES.
  const uses: Record<string, number> = {};
  const speakerFor: Record<string, string> = {};
  for (const code of sentenceCodes) {
    const candidates = shuffle(
      SPEAKERS.filter((s) => (uses[s] ?? 0) < MAX_SPEAKER_USES),
    );
    const speaker = candidates[0];
    if (!speaker) throw new Error("ran out of speakers under MAX_SPEAKER_USES");
    uses[speaker] = (uses[speaker] ?? 0) + 1;
    speakerFor[code] = speaker;
  }

  // 4. Load the clips needed for those (speaker, sentence) pairs.
  const usedSpeakers = [...new Set(Object.values(speakerFor))];
  const { data: clips, error: cErr } = await supa
    .from("survey_clips")
    .select("id, student_id, sentence_code, condition, sentence, storage_path")
    .in("student_id", usedSpeakers)
    .in("sentence_code", sentenceCodes);
  if (cErr) throw new Error(`load clips: ${cErr.message}`);
  const clipMap = new Map<string, ClipRow>();
  for (const c of (clips ?? []) as ClipRow[]) {
    clipMap.set(`${c.student_id}|${c.sentence_code}|${c.condition}`, c);
  }
  const getClip = (speaker: string, code: string, cond: string) => {
    const c = clipMap.get(`${speaker}|${code}|${cond}`);
    if (!c) throw new Error(`missing clip ${speaker} ${code} ${cond}`);
    return c;
  };

  // 5. Build the 18 comparison items.
  const comparisons: Record<string, unknown>[] = [];
  for (const code of sentenceCodes) {
    const speaker = speakerFor[code];
    for (const q of COMPARISON_QUESTIONS) {
      const [x, y] = shuffle([...q.pair]); // randomise which condition is A
      const a = getClip(speaker, code, x);
      const b = getClip(speaker, code, y);
      comparisons.push({
        participant_id: participantId,
        item_type: "comparison",
        sentence_code: code,
        student_id: speaker,
        question_type: q.type,
        clip_a_id: a.id,
        clip_b_id: b.id,
        condition_a: x,
        condition_b: y,
      });
    }
  }
  shuffle(comparisons);

  // 6. Insert the attention checks in the MIDDLE of the shuffled comparisons
  //    (≈ 1/3 and 2/3 through), never at the very start or end.
  const attention = ATTENTION_CHECKS.map((a) => ({
    participant_id: participantId,
    item_type: "attention",
    expected: a.expected,
  }));
  const seq: Record<string, unknown>[] = [...comparisons];
  const p1 = 5 + Math.floor(Math.random() * 4); // index 5..8
  seq.splice(p1, 0, attention[0]);
  const p2 = 11 + Math.floor(Math.random() * 4); // index 11..14
  seq.splice(p2, 0, attention[1] ?? attention[0]);
  const ordered = seq.map((row, i) => ({ ...row, order_index: i + 1 }));

  const { error: iErr } = await supa.from("survey_items").insert(ordered);
  if (iErr) throw new Error(`insert items: ${iErr.message}`);

  // 7. Read back via the same join shape.
  return getOrCreateSurveyItems(participantId);
}

type ItemJoinRow = {
  id: string;
  order_index: number;
  item_type: string;
  answered_at: string | null;
  question_type: string | null;
  expected: number | null;
  sentence_code: string | null;
  clip_a: { storage_path: string; sentence: string | null } | { storage_path: string; sentence: string | null }[] | null;
  clip_b: { storage_path: string } | { storage_path: string }[] | null;
};

function one<T>(v: T | T[] | null): T | null {
  return Array.isArray(v) ? (v[0] ?? null) : v;
}

function rowToItem(row: ItemJoinRow): SurveyItem {
  if (row.item_type === "attention") {
    // Match the prompt by its expected value (position is no longer fixed).
    const check = ATTENTION_CHECKS.find((c) => c.expected === row.expected);
    return {
      item_id: row.id,
      order_index: row.order_index,
      item_type: "attention",
      answered: row.answered_at != null,
      prompt: check?.prompt ?? "請依指示作答。",
    };
  }
  const a = one(row.clip_a);
  const b = one(row.clip_b);
  return {
    item_id: row.id,
    order_index: row.order_index,
    item_type: "comparison",
    answered: row.answered_at != null,
    question_type: row.question_type ?? undefined,
    sentence: a?.sentence ?? null,
    clip_a_url: a?.storage_path,
    clip_b_url: b?.storage_path,
  };
}
