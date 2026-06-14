// /api/finalize-voice
//   POST  { participant_id }      → kicks off Fish IVC + ElevenLabs IVC
//                                   (synchronous; both APIs typically respond in seconds)
//                                   stores voice_ids on research_participants
//   GET   ?id=<participant_id>    → { fish_ready, elevenlabs_ready, participant_status }
//                                   used by /processing page to poll
//
// External APIs used:
//   - Fish Audio   POST https://api.fish.audio/model         (creates voice model)
//   - ElevenLabs   POST https://api.elevenlabs.io/v1/voices/add (Instant Voice Cloning)
//
// Required env (server-only):
//   FISH_API_KEY                  (not present yet → STOP-and-report)
//   ELEVENLABS_API_KEY            (present in .env.local)
//   SUPABASE_SERVICE_ROLE_KEY     (not present yet → STOP-and-report)

import { NextResponse } from "next/server";
import { researchServiceClient } from "@/lib/supabase/research";

const FISH_API_URL = "https://api.fish.audio/model";
const ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/voices/add";

export async function GET(req: Request) {
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return errorResponse(503, "SUPABASE_SERVICE_ROLE_KEY is not set on the server.");
  }
  const url = new URL(req.url);
  const participantId = (url.searchParams.get("id") ?? "").trim();
  if (!/^[0-9a-f-]{36}$/i.test(participantId)) {
    return errorResponse(400, "Invalid id.");
  }

  const supa = researchServiceClient();
  const { data, error } = await supa
    .from("research_participants")
    .select("fish_voice_id, elevenlabs_voice_id, status")
    .eq("id", participantId)
    .maybeSingle();

  if (error) return errorResponse(500, `Database error: ${error.message}`);
  if (!data) return errorResponse(404, "Participant not found.");

  return NextResponse.json({
    fish_ready: Boolean(data.fish_voice_id),
    elevenlabs_ready: Boolean(data.elevenlabs_voice_id),
    participant_status: data.status,
  });
}

export async function POST(req: Request) {
  const missing: string[] = [];
  if (!process.env.SUPABASE_SERVICE_ROLE_KEY) missing.push("SUPABASE_SERVICE_ROLE_KEY");
  if (!process.env.FISH_API_KEY) missing.push("FISH_API_KEY");
  if (!process.env.ELEVENLABS_API_KEY) missing.push("ELEVENLABS_API_KEY");
  if (missing.length > 0) {
    return errorResponse(
      503,
      `Server is missing required env vars: ${missing.join(", ")}. Voice cloning cannot proceed.`,
    );
  }

  let body: { participant_id?: unknown };
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Body must be JSON.");
  }
  const participantId =
    typeof body.participant_id === "string" ? body.participant_id.trim() : "";
  if (!/^[0-9a-f-]{36}$/i.test(participantId)) {
    return errorResponse(400, "Invalid participant_id.");
  }

  const supa = researchServiceClient();

  // 1. Confirm participant + load existing voice ids (idempotent retry).
  const { data: participant, error: pErr } = await supa
    .from("research_participants")
    .select("id, fish_voice_id, elevenlabs_voice_id, external_code")
    .eq("id", participantId)
    .maybeSingle();
  if (pErr) return errorResponse(500, `Database error: ${pErr.message}`);
  if (!participant) return errorResponse(404, "Participant not found.");

  // 2. Fetch the 5 calibration recordings from Storage.
  const { data: recordings, error: rErr } = await supa
    .from("research_recordings")
    .select("storage_path, mime_type")
    .eq("participant_id", participantId)
    .like("storage_path", `${participantId}/calibration/%`)
    .order("storage_path");
  if (rErr) return errorResponse(500, `Database error: ${rErr.message}`);
  if (!recordings || recordings.length < 5) {
    return errorResponse(
      409,
      `Need 5 calibration recordings; only ${recordings?.length ?? 0} present.`,
    );
  }

  const blobs: { path: string; blob: Blob; mime: string }[] = [];
  for (const r of recordings) {
    const { data: file, error: dErr } = await supa.storage
      .from("research_recordings")
      .download(r.storage_path as string);
    if (dErr || !file) {
      return errorResponse(
        500,
        `Could not download ${r.storage_path}: ${dErr?.message ?? "unknown"}`,
      );
    }
    blobs.push({
      path: r.storage_path as string,
      blob: file,
      mime: (r.mime_type as string) ?? "audio/webm",
    });
  }

  // 3. Run Fish IVC and ElevenLabs IVC in parallel (skip whichever is already done).
  const fishPromise = participant.fish_voice_id
    ? Promise.resolve(participant.fish_voice_id as string)
    : createFishVoice(blobs, participant.external_code as string);
  const elevenPromise = participant.elevenlabs_voice_id
    ? Promise.resolve(participant.elevenlabs_voice_id as string)
    : createElevenLabsVoice(blobs, participant.external_code as string);

  const [fishResult, elevenResult] = await Promise.allSettled([
    fishPromise,
    elevenPromise,
  ]);

  const updates: Record<string, string> = {};
  let fishVoiceId: string | null = null;
  let elevenVoiceId: string | null = null;
  const errors: string[] = [];

  if (fishResult.status === "fulfilled") {
    fishVoiceId = fishResult.value;
    if (!participant.fish_voice_id) updates.fish_voice_id = fishVoiceId;
  } else {
    errors.push(`Fish: ${formatErr(fishResult.reason)}`);
  }

  if (elevenResult.status === "fulfilled") {
    elevenVoiceId = elevenResult.value;
    if (!participant.elevenlabs_voice_id) updates.elevenlabs_voice_id = elevenVoiceId;
  } else {
    errors.push(`ElevenLabs: ${formatErr(elevenResult.reason)}`);
  }

  if (Object.keys(updates).length > 0) {
    const { error: uErr } = await supa
      .from("research_participants")
      .update(updates)
      .eq("id", participantId);
    if (uErr) {
      return errorResponse(500, `Could not save voice ids: ${uErr.message}`);
    }
  }

  // Audit event
  await supa.from("research_events").insert({
    participant_id: participantId,
    event_type: errors.length === 0 ? "session_completed" : "error",
    event_payload: {
      kind: "finalize_voice",
      fish_ready: Boolean(fishVoiceId),
      elevenlabs_ready: Boolean(elevenVoiceId),
      errors,
    },
    client_ts: new Date().toISOString(),
  });

  if (errors.length > 0 && (!fishVoiceId || !elevenVoiceId)) {
    return NextResponse.json(
      {
        fish_voice_id: fishVoiceId,
        elevenlabs_voice_id: elevenVoiceId,
        errors,
      },
      { status: 502 },
    );
  }

  return NextResponse.json({
    fish_voice_id: fishVoiceId,
    elevenlabs_voice_id: elevenVoiceId,
  });
}

// ──────────────────────────────────────────────────────────────────────
// External API integrations
// ──────────────────────────────────────────────────────────────────────

async function createFishVoice(
  blobs: { path: string; blob: Blob; mime: string }[],
  externalCode: string,
): Promise<string> {
  const fd = new FormData();
  fd.append("title", `shadow-${externalCode}`);
  fd.append(
    "description",
    "Voice model trained on five Mandarin calibration recordings for the Shadow Your Perfect Self study.",
  );
  fd.append("visibility", "private");
  fd.append("type", "tts");
  fd.append("train_mode", "fast"); // required; "fast" suits N=20 study (cross-lingual still works)
  for (const b of blobs) {
    // Fish accepts multiple voice samples under field name "voices".
    fd.append("voices", b.blob, b.path.split("/").pop() ?? "sample.webm");
  }

  const res = await fetch(FISH_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.FISH_API_KEY}`,
    },
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Fish ${res.status}: ${text.slice(0, 300)}`);
  }
  const json = (await res.json()) as { _id?: string; id?: string };
  const voiceId = json._id ?? json.id;
  if (!voiceId) throw new Error("Fish response missing voice id.");
  return voiceId;
}

async function createElevenLabsVoice(
  blobs: { path: string; blob: Blob; mime: string }[],
  externalCode: string,
): Promise<string> {
  const fd = new FormData();
  fd.append("name", `shadow-${externalCode}`);
  fd.append(
    "description",
    "Voice clone for Shadow Your Perfect Self study. Created from five Mandarin samples.",
  );
  for (const b of blobs) {
    fd.append("files", b.blob, b.path.split("/").pop() ?? "sample.webm");
  }

  const res = await fetch(ELEVENLABS_API_URL, {
    method: "POST",
    headers: {
      "xi-api-key": process.env.ELEVENLABS_API_KEY!,
    },
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ElevenLabs ${res.status}: ${text.slice(0, 300)}`);
  }
  const json = (await res.json()) as { voice_id?: string };
  if (!json.voice_id) throw new Error("ElevenLabs response missing voice_id.");
  return json.voice_id;
}

function formatErr(reason: unknown): string {
  if (reason instanceof Error) return reason.message;
  return String(reason);
}

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}
