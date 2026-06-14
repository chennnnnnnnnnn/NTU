// POST /api/recording-upload
//
// Unified multipart upload for the pre / train / post stages.
// (Phase D refactor: replaces /api/pretest-upload, which is left in place
// for now to avoid breaking the still-deployed flow.)
//
// Fields:
//   participant_id   uuid                          required
//   assignment_id    uuid                          required
//   stimulus_code    string                        required (validated against allowed set)
//   test_stage       "pre" | "train" | "post"      required (must match assignment row)
//   trial_n          integer                       required for train, ignored otherwise
//   duration_sec     numeric                       required
//   audio            Blob (audio/webm|mp4|wav|ogg) required
//
// Returns: { storage_path }

import { NextResponse } from "next/server";
import { researchServiceClient } from "@/lib/supabase/research";

const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED_MIME = new Set([
  "audio/webm",
  "audio/webm;codecs=opus",
  "audio/mp4",
  "audio/mp4;codecs=mp4a.40.2",
  "audio/wav",
  "audio/ogg",
]);
const ID_RE = /^[0-9a-f-]{36}$/i;
const CODE_RE = /^(alpha|beta|gamma)[1-6]$/;
const STAGE_RE = /^(pre|train|post)$/;

export async function POST(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return errorResponse(400, "Body must be multipart/form-data.");
  }

  const participantId = String(form.get("participant_id") ?? "").trim();
  const assignmentId = String(form.get("assignment_id") ?? "").trim();
  const stimulusCode = String(form.get("stimulus_code") ?? "").trim();
  const testStage = String(form.get("test_stage") ?? "").trim();
  const trialRaw = String(form.get("trial_n") ?? "").trim();
  const durationRaw = String(form.get("duration_sec") ?? "").trim();
  const attemptRaw = String(form.get("attempt_n") ?? "").trim();
  const listenRaw = String(form.get("listen_count") ?? "").trim();
  const audio = form.get("audio");

  const attemptN = Number.parseInt(attemptRaw, 10);
  const listenCount = Number.parseInt(listenRaw, 10);

  if (!ID_RE.test(participantId)) return errorResponse(400, "Invalid participant_id.");
  if (!ID_RE.test(assignmentId)) return errorResponse(400, "Invalid assignment_id.");
  if (!CODE_RE.test(stimulusCode)) return errorResponse(400, "Invalid stimulus_code.");
  if (!STAGE_RE.test(testStage)) return errorResponse(400, "test_stage must be pre, train, or post.");
  const durationSec = Number.parseFloat(durationRaw);
  if (!Number.isFinite(durationSec) || durationSec <= 0) {
    return errorResponse(400, "duration_sec must be a positive number.");
  }
  let trialN: number | null = null;
  if (testStage === "train") {
    trialN = Number.parseInt(trialRaw, 10);
    if (!Number.isFinite(trialN) || trialN < 1 || trialN > 99) {
      return errorResponse(400, "trial_n must be 1..99 for train stage.");
    }
  }
  if (!(audio instanceof Blob)) return errorResponse(400, "audio field must be a Blob.");
  if (audio.size === 0) return errorResponse(400, "audio blob is empty.");
  if (audio.size > MAX_BYTES) return errorResponse(413, `audio exceeds ${MAX_BYTES} bytes.`);
  const mime = (audio.type || "application/octet-stream").toLowerCase();
  if (!ALLOWED_MIME.has(mime) && !ALLOWED_MIME.has(mime.split(";")[0])) {
    return errorResponse(415, `Unsupported audio mime: ${mime}`);
  }

  const supa = researchServiceClient();

  const { data: assignment, error: aErr } = await supa
    .from("research_assignments")
    .select("id, participant_id, test_stage")
    .eq("id", assignmentId)
    .maybeSingle();
  if (aErr) return errorResponse(500, `Database error: ${aErr.message}`);
  if (!assignment) return errorResponse(404, "Assignment not found.");
  if (assignment.participant_id !== participantId) {
    return errorResponse(403, "Assignment does not belong to this participant.");
  }
  if (assignment.test_stage !== testStage) {
    return errorResponse(
      409,
      `Assignment test_stage is ${assignment.test_stage}, expected ${testStage}.`,
    );
  }

  // Storage path: <participant_id>/<stage>/<stimulus_code>[_t<n>].<ext>
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("wav") ? "wav" : "webm";
  const fname =
    testStage === "train" && trialN != null
      ? `${stimulusCode}_t${String(trialN).padStart(2, "0")}.${ext}`
      : `${stimulusCode}.${ext}`;
  const storagePath = `${participantId}/${testStage}/${fname}`;

  const arrayBuffer = await audio.arrayBuffer();
  const { error: upErr } = await supa.storage
    .from("research_recordings")
    .upload(storagePath, arrayBuffer, {
      contentType: mime,
      upsert: true,
    });
  if (upErr) return errorResponse(500, `Storage upload failed: ${upErr.message}`);

  // Replace any prior row at this exact path (re-record support).
  await supa
    .from("research_recordings")
    .delete()
    .eq("participant_id", participantId)
    .eq("storage_path", storagePath);

  const { error: recErr } = await supa.from("research_recordings").insert({
    participant_id: participantId,
    assignment_id: assignmentId,
    storage_bucket: "research_recordings",
    storage_path: storagePath,
    mime_type: mime,
    duration_sec: durationSec,
    client_recorded_at: new Date().toISOString(),
    trial_n: trialN,
  });
  if (recErr) return errorResponse(500, `Could not record metadata: ${recErr.message}`);

  await supa.from("research_events").insert({
    participant_id: participantId,
    event_type: "upload_success",
    event_payload: {
      kind: testStage,
      stimulus_code: stimulusCode,
      assignment_id: assignmentId,
      trial_n: trialN,
      bytes: audio.size,
      mime,
      duration_sec: durationSec,
      attempt_n: Number.isFinite(attemptN) ? attemptN : null,
      listen_count: Number.isFinite(listenCount) ? listenCount : null,
    },
    client_ts: new Date().toISOString(),
  });

  return NextResponse.json({ storage_path: storagePath });
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
