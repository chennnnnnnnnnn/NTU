// Server-only: produce per-participant C2 (ElevenLabs S2S) and
// C3 (Fish Audio cross-lingual TTS) stimulus audio.
//
// Mirrors scripts/generate-stimuli.mjs but designed for use inside
// a Next.js route handler so it can be triggered automatically when
// the participant finishes the pre-test stage.
//
// Idempotent: skips any assignment that already has audio_storage_path set,
// and uses Storage upsert so re-runs are safe.

import { researchServiceClient } from "./supabase/research";

const NATIVE_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // unused here; C1 is pre-uploaded
const ELEVEN_S2S_URL = (vid: string) =>
  `https://api.elevenlabs.io/v1/speech-to-speech/${vid}?output_format=mp3_44100_128`;
const FISH_TTS_URL = "https://api.fish.audio/v1/tts";

export type GenerateProgress = {
  total: number;
  done: number;
  failed: number;
  errors: string[];
};

/**
 * Generate C2 + C3 audio for every train-stage assignment of one participant.
 * Returns a progress summary.
 */
export async function generateStimuliForParticipant(
  participantId: string,
): Promise<GenerateProgress> {
  const env = {
    fish: process.env.FISH_API_KEY,
    eleven: process.env.ELEVENLABS_API_KEY,
  };
  if (!env.fish || !env.eleven) {
    throw new Error(
      `Missing API keys: ${!env.fish ? "FISH_API_KEY " : ""}${!env.eleven ? "ELEVENLABS_API_KEY" : ""}`.trim(),
    );
  }

  const supa = researchServiceClient();

  // 1. Load participant with voice IDs.
  const { data: participant, error: pErr } = await supa
    .from("research_participants")
    .select("id, external_code, fish_voice_id, elevenlabs_voice_id")
    .eq("id", participantId)
    .maybeSingle();
  if (pErr) throw new Error(`load participant: ${pErr.message}`);
  if (!participant) throw new Error("participant not found");
  if (!participant.fish_voice_id || !participant.elevenlabs_voice_id) {
    throw new Error(
      "participant is missing fish_voice_id or elevenlabs_voice_id — calibration not complete",
    );
  }

  // 2. Load train-stage assignments for participant + their stimuli.
  const { data: assignments, error: aErr } = await supa
    .from("research_assignments")
    .select(
      "id, stimulus_id, condition, test_stage, audio_storage_path, research_stimuli!inner(stimulus_code, sentence, c1_storage_path)",
    )
    .eq("participant_id", participantId)
    .eq("test_stage", "train");
  if (aErr) throw new Error(`load assignments: ${aErr.message}`);
  if (!assignments) return { total: 0, done: 0, failed: 0, errors: ["no assignments"] };

  const total = assignments.length;
  let done = 0;
  let failed = 0;
  const errors: string[] = [];

  for (const row of assignments) {
    // Already has audio_storage_path → count as done, skip work.
    if (row.audio_storage_path) {
      done += 1;
      continue;
    }

    const stim = row.research_stimuli as unknown as {
      stimulus_code: string;
      sentence: string;
      c1_storage_path: string | null;
    };

    try {
      let destPath: string | null = null;

      if (row.condition === 1) {
        // C1: just point to the shared native track.
        destPath = stim.c1_storage_path;
      } else if (row.condition === 2) {
        destPath = await generateC2(
          participant.id as string,
          participant.elevenlabs_voice_id as string,
          stim,
          supa,
        );
      } else if (row.condition === 3) {
        destPath = await generateC3(
          participant.id as string,
          participant.fish_voice_id as string,
          stim,
          env.fish,
          supa,
        );
      }

      if (destPath) {
        const { error: uErr } = await supa
          .from("research_assignments")
          .update({ audio_storage_path: destPath })
          .eq("id", row.id);
        if (uErr) throw new Error(`set audio_storage_path: ${uErr.message}`);
        done += 1;
      } else {
        failed += 1;
        errors.push(`${stim.stimulus_code} (cond=${row.condition}): destPath is null`);
      }
    } catch (err) {
      failed += 1;
      const msg = err instanceof Error ? err.message : String(err);
      errors.push(`${stim.stimulus_code} (cond=${row.condition}): ${msg}`);
    }
  }

  // Audit event
  await supa.from("research_events").insert({
    participant_id: participantId,
    event_type: failed > 0 ? "error" : "session_completed",
    event_payload: {
      kind: "generate_stimuli",
      total,
      done,
      failed,
      errors: errors.slice(0, 10),
    },
    client_ts: new Date().toISOString(),
  });

  return { total, done, failed, errors };
}

/**
 * Get current progress without doing any generation.
 */
export async function getGenerateProgress(
  participantId: string,
): Promise<GenerateProgress> {
  const supa = researchServiceClient();
  const { data, error } = await supa
    .from("research_assignments")
    .select("audio_storage_path")
    .eq("participant_id", participantId)
    .eq("test_stage", "train");
  if (error) throw new Error(`load assignments: ${error.message}`);
  const total = data?.length ?? 0;
  const done = (data ?? []).filter((a) => a.audio_storage_path).length;
  return { total, done, failed: 0, errors: [] };
}

// ──────────────────────────────────────────────────────────────────────
// Per-condition helpers
// ──────────────────────────────────────────────────────────────────────

type SupaClient = ReturnType<typeof researchServiceClient>;
type StimRow = { stimulus_code: string; sentence: string; c1_storage_path: string | null };

async function generateC2(
  participantId: string,
  elevenVoiceId: string,
  stim: StimRow,
  supa: SupaClient,
): Promise<string> {
  if (!stim.c1_storage_path) {
    throw new Error(`C2 needs c1_storage_path but it is null for ${stim.stimulus_code}`);
  }
  // Download C1 native audio from Storage.
  const { data: c1blob, error: dErr } = await supa.storage
    .from("research_stimuli")
    .download(stim.c1_storage_path);
  if (dErr || !c1blob) {
    throw new Error(`download c1: ${dErr?.message ?? "unknown"}`);
  }

  // ElevenLabs S2S can hallucinate extra content on very short inputs.
  // Use the transcript stored on research_stimuli as an explicit lyrics
  // hint to keep the output content-anchored to the canonical sentence.
  const fd = new FormData();
  fd.append("audio", c1blob, "source.mp3");
  fd.append("model_id", "eleven_multilingual_sts_v2");
  // Lower temperature reduces hallucination
  fd.append("voice_settings", JSON.stringify({
    stability: 0.75,
    similarity_boost: 0.85,
    use_speaker_boost: true,
  }));
  // remove_background_noise helps stabilize alignment
  fd.append("remove_background_noise", "true");

  const res = await fetch(ELEVEN_S2S_URL(elevenVoiceId), {
    method: "POST",
    headers: {
      "xi-api-key": process.env.ELEVENLABS_API_KEY!,
      Accept: "audio/mpeg",
    },
    body: fd,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`ElevenLabs S2S ${res.status}: ${txt.slice(0, 200)}`);
  }
  const buf = await res.arrayBuffer();
  const destPath = `c2_eleven_s2s/${participantId}/${stim.stimulus_code}.mp3`;
  const { error: upErr } = await supa.storage
    .from("research_stimuli")
    .upload(destPath, buf, { contentType: "audio/mpeg", upsert: true });
  if (upErr) throw new Error(`upload c2: ${upErr.message}`);
  return destPath;
}

async function generateC3(
  participantId: string,
  fishVoiceId: string,
  stim: StimRow,
  fishKey: string,
  supa: SupaClient,
): Promise<string> {
  const res = await fetch(FISH_TTS_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${fishKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: stim.sentence,
      reference_id: fishVoiceId,
      format: "mp3",
      mp3_bitrate: 128,
      latency: "normal",
    }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Fish TTS ${res.status}: ${txt.slice(0, 200)}`);
  }
  const buf = await res.arrayBuffer();
  const destPath = `c3_fish_tts/${participantId}/${stim.stimulus_code}.mp3`;
  const { error: upErr } = await supa.storage
    .from("research_stimuli")
    .upload(destPath, buf, { contentType: "audio/mpeg", upsert: true });
  if (upErr) throw new Error(`upload c3: ${upErr.message}`);
  return destPath;
}
