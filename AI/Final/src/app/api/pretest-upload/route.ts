// POST /api/pretest-upload
// multipart/form-data:
//   participant_id, assignment_id, stimulus_code, duration_sec, audio (blob)
// Returns: { storage_path }

import { NextResponse } from "next/server";
import { researchServiceClient } from "@/lib/supabase/research";

const MAX_BYTES = 5 * 1024 * 1024; // 5 MB (pre-test clips are ≤20 s)
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
  const durationRaw = String(form.get("duration_sec") ?? "").trim();
  const attemptRaw = String(form.get("attempt_n") ?? "").trim();
  const attemptN = Number.parseInt(attemptRaw, 10);
  const audio = form.get("audio");

  if (!ID_RE.test(participantId)) return errorResponse(400, "Invalid participant_id.");
  if (!ID_RE.test(assignmentId)) return errorResponse(400, "Invalid assignment_id.");
  if (!CODE_RE.test(stimulusCode)) return errorResponse(400, "Invalid stimulus_code.");
  const durationSec = Number.parseFloat(durationRaw);
  if (!Number.isFinite(durationSec) || durationSec <= 0) {
    return errorResponse(400, "duration_sec must be a positive number.");
  }
  if (!(audio instanceof Blob)) return errorResponse(400, "audio field must be a Blob.");
  if (audio.size === 0) return errorResponse(400, "audio blob is empty.");
  if (audio.size > MAX_BYTES) return errorResponse(413, `audio exceeds ${MAX_BYTES} bytes.`);
  const mime = (audio.type || "application/octet-stream").toLowerCase();
  if (!ALLOWED_MIME.has(mime) && !ALLOWED_MIME.has(mime.split(";")[0])) {
    return errorResponse(415, `Unsupported audio mime: ${mime}`);
  }

  const supa = researchServiceClient();

  // Verify the assignment exists, belongs to participant, and is pre-stage.
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
  if (assignment.test_stage !== "pre") {
    return errorResponse(409, `Assignment test_stage is ${assignment.test_stage}, expected pre.`);
  }

  // Storage path: <participant_id>/pre/<stimulus_code>.<ext>
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("wav") ? "wav" : "webm";
  const storagePath = `${participantId}/pre/${stimulusCode}.${ext}`;

  const arrayBuffer = await audio.arrayBuffer();
  const { error: upErr } = await supa.storage
    .from("research_recordings")
    .upload(storagePath, arrayBuffer, {
      contentType: mime,
      upsert: true,
    });
  if (upErr) {
    return errorResponse(500, `Storage upload failed: ${upErr.message}`);
  }

  // Replace any prior recording row for the same path (re-record support).
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
  });
  if (recErr) {
    return errorResponse(500, `Could not record metadata: ${recErr.message}`);
  }

  // Audit
  await supa.from("research_events").insert({
    participant_id: participantId,
    event_type: "upload_success",
    event_payload: {
      kind: "pretest",
      stimulus_code: stimulusCode,
      assignment_id: assignmentId,
      bytes: audio.size,
      mime,
      duration_sec: durationSec,
      attempt_n: Number.isFinite(attemptN) ? attemptN : null,
    },
    client_ts: new Date().toISOString(),
  });

  return NextResponse.json({ storage_path: storagePath });
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
