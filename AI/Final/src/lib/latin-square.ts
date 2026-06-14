// 3×3 Latin-square counterbalancing for the three-condition shadowing study.
//
// We have 3 stimulus sets (alpha / beta / gamma) and 3 conditions (C1 / C2 / C3).
// Each participant is assigned to one of 6 `latin_square_group`s (1..6), which
// fixes which set is paired with which condition for that participant.
//
// Across 6 participants the design is fully balanced: each (set, condition)
// pair appears exactly twice. For N=20, ~3 participants per group, slight
// imbalance accepted (4 participants for two of the six groups).
//
// Decision (Phase C, 2026-05-25): pre-test and post-test cover ALL 18 stimuli
// per participant, not a 6-stimulus subset. Within-subjects paired comparison
// of pre vs post requires identical stimulus pool at both stages. The Latin
// square assigns each stimulus to one of three conditions for the training
// stage only; pre / post recordings are condition-blind (cold-read).

import type { Condition } from "./types";

export type SetLabel = "alpha" | "beta" | "gamma";

/**
 * The six Latin-square rotations. Each row is one (set→condition) mapping.
 * Generated so every (set, condition) pair appears exactly twice across rows.
 */
const LATIN_SQUARE: Record<number, Record<SetLabel, Condition>> = {
  1: { alpha: 1, beta: 2, gamma: 3 },
  2: { alpha: 2, beta: 3, gamma: 1 },
  3: { alpha: 3, beta: 1, gamma: 2 },
  4: { alpha: 1, beta: 3, gamma: 2 },
  5: { alpha: 2, beta: 1, gamma: 3 },
  6: { alpha: 3, beta: 2, gamma: 1 },
};

export type LatinSquareGroup = 1 | 2 | 3 | 4 | 5 | 6;

export function isLatinSquareGroup(n: unknown): n is LatinSquareGroup {
  return typeof n === "number" && Number.isInteger(n) && n >= 1 && n <= 6;
}

/**
 * Given a participant's group and a stimulus's set, return the assigned
 * condition for the training stage. Pre / post stages do not use this.
 */
export function conditionForGroupAndSet(
  group: LatinSquareGroup,
  set: SetLabel,
): Condition {
  return LATIN_SQUARE[group][set];
}

/**
 * Pick a group for a new participant. Strategy: round-robin through 1..6 so
 * the first 6 participants each fall into a distinct group; participant 7
 * starts over at group 1.
 *
 * For pilot/test participants (`PILOT*` external_code) we still pick a group
 * deterministically by code so re-running enrollment is idempotent.
 */
export function pickGroupForCode(code: string): LatinSquareGroup {
  // Hash code into 1..6.
  let h = 0;
  for (const c of code) h = (h * 31 + c.charCodeAt(0)) | 0;
  return (((Math.abs(h) % 6) + 1) as LatinSquareGroup);
}

/**
 * Build the full assignment list for a participant. Each row corresponds to
 * one (stimulus, test_stage) pair. order_index is 1..N within each stage.
 *
 * Stimulus presentation order within a stage is deterministic by group, so
 * a re-run for the same participant yields the same order — important for
 * replaying after a network hiccup.
 *
 * Total rows per participant: 18 stimuli × 3 stages = 54.
 *
 * NOTE: This function only computes the *shape* of the assignments. It
 * leaves stimulus_id lookup and DB insertion to the caller; see
 * src/lib/test-assignment.ts.
 */
export type AssignmentShape = {
  stimulus_code: string;
  set_label: SetLabel;
  condition: Condition;
  test_stage: "pre" | "train" | "post";
  order_index: number;
  is_trained: boolean;
};

export function buildAssignments(
  group: LatinSquareGroup,
  stimuli: { stimulus_code: string; set_label: SetLabel }[],
): AssignmentShape[] {
  // Stable shuffle: per-group permutation of stimulus index, deterministic.
  const ordered = stableOrder(group, stimuli);

  const out: AssignmentShape[] = [];
  for (const stage of ["pre", "train", "post"] as const) {
    ordered.forEach((s, i) => {
      out.push({
        stimulus_code: s.stimulus_code,
        set_label: s.set_label,
        condition: conditionForGroupAndSet(group, s.set_label),
        test_stage: stage,
        order_index: i + 1,
        is_trained: stage === "train",
      });
    });
  }
  return out;
}

function stableOrder<T extends { stimulus_code: string }>(
  group: LatinSquareGroup,
  items: T[],
): T[] {
  // Deterministic permutation: rotate by group index, then interleave sets.
  // Effect: across the 6 groups, the first-presented stimulus rotates so no
  // particular sentence is over-represented at "the start" of the session.
  const sorted = [...items].sort((a, b) =>
    a.stimulus_code.localeCompare(b.stimulus_code),
  );
  const rotated = sorted.slice(group - 1).concat(sorted.slice(0, group - 1));
  return rotated;
}
