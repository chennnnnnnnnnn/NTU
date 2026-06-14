// Shared resume routing — pick the next stage URL for a participant id.
//
// Used by server-side pages to redirect users who land on a stage they've
// already passed (e.g. opening /calibration after they already finished
// pre-test). Mirrors the routing rules in /api/enroll.

import { researchServiceClient } from "@/lib/supabase/research";

export type ResumePoint =
  | { path: "/calibration"; reason: "needs_calibration" }
  | { path: "/processing"; reason: "voice_clones_pending" }
  | { path: "/pre-test"; reason: "needs_pre_test" }
  | { path: "/train"; reason: "needs_train" }
  | { path: "/post-test"; reason: "needs_post_test" }
  | { path: "/done"; reason: "needs_survey" }
  | { path: "/done/complete"; reason: "all_done" }
  | { path: "/"; reason: "withdrawn_or_unknown" };

/**
 * Read the participant's row and return where they should be.
 */
export async function resumePointFor(
  participantId: string,
): Promise<ResumePoint & { participantUrl: string }> {
  const supa = researchServiceClient();
  const { data, error } = await supa
    .from("research_participants")
    .select("status, fish_voice_id, elevenlabs_voice_id")
    .eq("id", participantId)
    .maybeSingle();

  if (error || !data) {
    return { path: "/", reason: "withdrawn_or_unknown", participantUrl: "/" };
  }

  const status = (data.status as string) ?? "enrolled";
  const fish = data.fish_voice_id as string | null;
  const eleven = data.elevenlabs_voice_id as string | null;

  let point: ResumePoint;
  switch (status) {
    case "pre_done":
      point = { path: "/train", reason: "needs_train" };
      break;
    case "train_done":
      point = { path: "/post-test", reason: "needs_post_test" };
      break;
    case "post_done":
      point = { path: "/done", reason: "needs_survey" };
      break;
    case "delayed_done":
      point = { path: "/done/complete", reason: "all_done" };
      break;
    case "withdrawn":
      point = { path: "/", reason: "withdrawn_or_unknown" };
      break;
    case "enrolled":
    default:
      if (fish && eleven) {
        // Calibration done + both voice clones minted → ready for pre-test.
        // Note: we keep status='enrolled' until /pre-test/done advances it
        // to 'pre_done'. The /processing page is a UX courtesy on the first
        // visit, not a required stage to gate on.
        point = { path: "/pre-test", reason: "needs_pre_test" };
      } else {
        point = { path: "/calibration", reason: "needs_calibration" };
      }
      break;
  }

  const url = point.path === "/" ? "/" : `${point.path}?id=${participantId}`;
  return { ...point, participantUrl: url };
}

/**
 * Stage check helper: returns the URL to redirect to if the participant
 * should NOT be on the current stage, or null if they're in the right place.
 */
export async function shouldRedirectFrom(
  participantId: string,
  currentStage: ResumePoint["path"],
): Promise<string | null> {
  const r = await resumePointFor(participantId);
  if (r.path === currentStage) return null;
  return r.participantUrl;
}
