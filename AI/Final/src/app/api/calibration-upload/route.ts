// POST /api/calibration-upload
// Body: multipart/form-data
//   participant_id (string)
//   segment_n      (string, 1..5)
//   duration_sec   (string, numeric)
//   audio          (Blob — audio/webm or audio/mp4)
// Returns: { storage_path }

import { NextResponse } from "next/server";
import { researchServiceClient } from "@/lib/supabase/research";

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
const ALLOWED_MIME = new Set([
  "audio/webm",
  "audio/webm;codecs=opus",
  "audio/mp4",
  "audio/mp4;codecs=mp4a.40.2",
  "audio/wav",
  "audio/ogg",
]);

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
  const segmentNRaw = String(form.get("segment_n") ?? "").trim();
  const durationRaw = String(form.get("duration_sec") ?? "").trim();
  const attemptRaw = String(form.get("attempt_n") ?? "").trim();
  const attemptN = Number.parseInt(attemptRaw, 10);
  const audio = form.get("audio");

  if (!/^[0-9a-f-]{36}$/i.test(participantId)) {
    return errorResponse(400, "Invalid participant_id.");
  }
  const segmentN = Number.parseInt(segmentNRaw, 10);
  if (!Number.isFinite(segmentN) || segmentN < 1 || segmentN > 5) {
    return errorResponse(400, "segment_n must be 1..5.");
  }
  const durationSec = Number.parseFloat(durationRaw);
  if (!Number.isFinite(durationSec) || durationSec <= 0) {
    return errorResponse(400, "duration_sec must be a positive number.");
  }
  if (!(audio instanceof Blob)) {
    return errorResponse(400, "audio field must be a Blob.");
  }
  if (audio.size === 0) {
    return errorResponse(400, "audio blob is empty.");
  }
  if (audio.size > MAX_BYTES) {
    return errorResponse(413, `audio exceeds ${MAX_BYTES} bytes.`);
  }
  const mime = (audio.type || "application/octet-stream").toLowerCase();
  if (!ALLOWED_MIME.has(mime) && !ALLOWED_MIME.has(mime.split(";")[0])) {
    return errorResponse(415, `Unsupported audio mime: ${mime}`);
  }

  const supa = researchServiceClient();

  // Verify participant exists.
  const { data: participant, error: pErr } = await supa
    .from("research_participants")
    .select("id")
    .eq("id", participantId)
    .maybeSingle();
  if (pErr) return errorResponse(500, `Database error: ${pErr.message}`);
  if (!participant) return errorResponse(404, "Participant not found.");

  // Storage path: <participant_id>/calibration/segment-<n>.<ext>
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("wav") ? "wav" : "webm";
  const storagePath = `${participantId}/calibration/segment-${segmentN}.${ext}`;

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

  // Recording row — assignment_id NULL because calibration is not a test stage.
  // Discriminator (segment_n + calibration) lives in storage_path.
  // Replace any prior row for same participant + path so re-records don't accumulate.
  await supa
    .from("research_recordings")
    .delete()
    .eq("participant_id", participantId)
    .eq("storage_path", storagePath);

  const { error: recErr } = await supa.from("research_recordings").insert({
    participant_id: participantId,
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
      kind: "calibration",
      segment_n: segmentN,
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
