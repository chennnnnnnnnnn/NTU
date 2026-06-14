// Batch-generate the per-condition stimulus audio for the study.
//
// Three conditions:
//   C1 — native model voice; shared across all participants.
//        Generated via ElevenLabs TTS with a built-in native English voice
//        (Rachel: 21m00Tcm4TlvDq8ikWAM). Uploaded once.
//   C2 — participant's voice with native accent; per-participant.
//        Generated via ElevenLabs Voice Changer (Speech-to-Speech):
//        source = the C1 native audio for the same stimulus,
//        target voice = participant.elevenlabs_voice_id.
//   C3 — participant's voice with L1 (Mandarin) accent leak; per-participant.
//        Generated via Fish Audio cross-lingual TTS using
//        participant.fish_voice_id directly on the English text.
//
// Storage layout:
//   research_stimuli/
//     c1_native/<stimulus_code>.mp3
//     c2_eleven_s2s/<participant_id>/<stimulus_code>.mp3
//     c3_fish_tts/<participant_id>/<stimulus_code>.mp3
//
// Flags:
//   --only-c1                 Generate / upload the C1 native set only.
//   --participant <code>      Restrict C2/C3 generation to one external_code.
//                             Default: all participants with both voice_ids.
//   --skip-c2                 Skip ElevenLabs S2S step.
//   --skip-c3                 Skip Fish TTS step.
//   --dry-run                 Print plan without API calls.
//
// Run:
//   node --env-file=.env.local scripts/generate-stimuli.mjs --only-c1
//   node --env-file=.env.local scripts/generate-stimuli.mjs --participant PILOT01
//   node --env-file=.env.local scripts/generate-stimuli.mjs --dry-run
//
// Quota notes (ElevenLabs Starter $6/mo):
//   - TTS: ~50 chars per sentence × 18 = ~900 chars total. Well under 30k.
//   - S2S: counts audio seconds, 18 × ~8 s = ~150 s per participant.
//   - 10 custom-voice cap → for N=20 the researcher must delete-and-recreate
//     voices between participants (or upgrade Creator tier).

import { createClient } from "@supabase/supabase-js";
import { writeFile, mkdir } from "node:fs/promises";

const NATIVE_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // ElevenLabs "Rachel"
const ELEVEN_TTS_MODEL = "eleven_multilingual_v2";

const args = parseArgs(process.argv.slice(2));

const env = {
  SUPABASE_URL: must("NEXT_PUBLIC_SUPABASE_URL"),
  SERVICE_ROLE: must("SUPABASE_SERVICE_ROLE_KEY"),
  ELEVENLABS_API_KEY: must("ELEVENLABS_API_KEY"),
  FISH_API_KEY: process.env.FISH_API_KEY,
};

if (!args.skipC3 && !env.FISH_API_KEY) {
  console.error("FISH_API_KEY not set; pass --skip-c3 or fix env.");
  process.exit(1);
}

const supa = createClient(env.SUPABASE_URL, env.SERVICE_ROLE, {
  auth: { autoRefreshToken: false, persistSession: false },
});

// ──────────────────────────────────────────────────────────────────────
// Load stimuli + participants
// ──────────────────────────────────────────────────────────────────────

const { data: stimuli, error: sErr } = await supa
  .from("research_stimuli")
  .select("id, stimulus_code, sentence, c1_storage_path")
  .order("stimulus_code");
if (sErr || !stimuli) bail(`load stimuli: ${sErr?.message}`);
console.log(`Loaded ${stimuli.length} stimuli`);

let participants = [];
if (!args.onlyC1) {
  let q = supa
    .from("research_participants")
    .select("id, external_code, fish_voice_id, elevenlabs_voice_id")
    .not("fish_voice_id", "is", null)
    .not("elevenlabs_voice_id", "is", null);
  if (args.participant) q = q.eq("external_code", args.participant);
  const { data, error } = await q;
  if (error) bail(`load participants: ${error.message}`);
  participants = data ?? [];
  console.log(`Loaded ${participants.length} participants with both voice_ids`);
}

// ──────────────────────────────────────────────────────────────────────
// C1 — native voice via ElevenLabs TTS
// ──────────────────────────────────────────────────────────────────────

await mkdir("/tmp/stimuli/c1", { recursive: true });

for (const s of stimuli) {
  const dest = `c1_native/${s.stimulus_code}.mp3`;
  if (s.c1_storage_path === dest) {
    console.log(`  C1 ${s.stimulus_code}: already at ${dest}, skip`);
    continue;
  }
  if (args.dryRun) {
    console.log(`  C1 ${s.stimulus_code}: would TTS "${s.sentence.slice(0, 30)}..."`);
    continue;
  }
  console.log(`  C1 ${s.stimulus_code}: TTS native …`);
  const buf = await elevenLabsTTS(s.sentence);
  await writeFile(`/tmp/stimuli/c1/${s.stimulus_code}.mp3`, Buffer.from(buf));
  const { error: upErr } = await supa.storage
    .from("research_stimuli")
    .upload(dest, buf, { contentType: "audio/mpeg", upsert: true });
  if (upErr) bail(`upload ${dest}: ${upErr.message}`);
  const { error: dbErr } = await supa
    .from("research_stimuli")
    .update({ c1_storage_path: dest })
    .eq("id", s.id);
  if (dbErr) bail(`update c1_storage_path ${s.stimulus_code}: ${dbErr.message}`);
  console.log(`    → uploaded ${dest} (${buf.byteLength} B)`);
}

if (args.onlyC1) {
  console.log("--only-c1 specified, done.");
  process.exit(0);
}

// ──────────────────────────────────────────────────────────────────────
// Per-participant: C2 (ElevenLabs S2S) and C3 (Fish TTS)
// ──────────────────────────────────────────────────────────────────────

for (const p of participants) {
  console.log(`\n── ${p.external_code} (${p.id}) ──`);

  // Load assignments for this participant; we only need to upsert
  // audio_storage_path on the train rows.
  const { data: assignments, error: aErr } = await supa
    .from("research_assignments")
    .select("id, stimulus_id, condition, test_stage")
    .eq("participant_id", p.id)
    .eq("test_stage", "train");
  if (aErr) bail(`load assignments for ${p.external_code}: ${aErr.message}`);

  for (const stim of stimuli) {
    const assignment = assignments.find((a) => a.stimulus_id === stim.id);
    if (!assignment) {
      console.log(`  ${stim.stimulus_code}: no train assignment, skip`);
      continue;
    }
    const cond = assignment.condition;

    let destPath = null;
    if (cond === 1) {
      destPath = stim.c1_storage_path; // share native track
    } else if (cond === 2 && !args.skipC2) {
      destPath = await runC2(p, stim);
    } else if (cond === 3 && !args.skipC3) {
      destPath = await runC3(p, stim);
    } else {
      console.log(`  ${stim.stimulus_code} cond=${cond}: skipped`);
      continue;
    }

    if (destPath && !args.dryRun) {
      const { error: uErr } = await supa
        .from("research_assignments")
        .update({ audio_storage_path: destPath })
        .eq("id", assignment.id);
      if (uErr) console.warn(`    ⚠ failed to set audio_storage_path: ${uErr.message}`);
    }
  }
}

console.log("\nDone.");

// ──────────────────────────────────────────────────────────────────────
// External API helpers
// ──────────────────────────────────────────────────────────────────────

async function elevenLabsTTS(text) {
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${NATIVE_VOICE_ID}?output_format=mp3_44100_128`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "xi-api-key": env.ELEVENLABS_API_KEY,
      "Content-Type": "application/json",
      Accept: "audio/mpeg",
    },
    body: JSON.stringify({ text, model_id: ELEVEN_TTS_MODEL }),
  });
  if (!res.ok) bail(`ElevenLabs TTS ${res.status}: ${await res.text()}`);
  return res.arrayBuffer();
}

async function elevenLabsS2S(sourceBlob, targetVoiceId) {
  const url = `https://api.elevenlabs.io/v1/speech-to-speech/${targetVoiceId}?output_format=mp3_44100_128`;
  const fd = new FormData();
  fd.append("audio", new Blob([sourceBlob], { type: "audio/mpeg" }), "source.mp3");
  fd.append("model_id", "eleven_multilingual_sts_v2");
  const res = await fetch(url, {
    method: "POST",
    headers: { "xi-api-key": env.ELEVENLABS_API_KEY, Accept: "audio/mpeg" },
    body: fd,
  });
  if (!res.ok) bail(`ElevenLabs S2S ${res.status}: ${await res.text()}`);
  return res.arrayBuffer();
}

async function fishTTS(text, voiceId) {
  const url = "https://api.fish.audio/v1/tts";
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.FISH_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      reference_id: voiceId,
      format: "mp3",
      mp3_bitrate: 128,
      latency: "normal",
    }),
  });
  if (!res.ok) bail(`Fish TTS ${res.status}: ${await res.text()}`);
  return res.arrayBuffer();
}

async function runC2(participant, stim) {
  if (!stim.c1_storage_path) {
    console.log(`  ${stim.stimulus_code} C2: no c1 source yet, skip`);
    return null;
  }
  const dest = `c2_eleven_s2s/${participant.id}/${stim.stimulus_code}.mp3`;
  if (args.dryRun) {
    console.log(`  ${stim.stimulus_code} C2: would S2S → ${dest}`);
    return dest;
  }
  console.log(`  ${stim.stimulus_code} C2: ElevenLabs S2S …`);
  const { data: c1, error: dErr } = await supa.storage
    .from("research_stimuli")
    .download(stim.c1_storage_path);
  if (dErr || !c1) bail(`download c1 for ${stim.stimulus_code}: ${dErr?.message}`);
  const c1Buf = await c1.arrayBuffer();
  const buf = await elevenLabsS2S(c1Buf, participant.elevenlabs_voice_id);
  const { error: upErr } = await supa.storage
    .from("research_stimuli")
    .upload(dest, buf, { contentType: "audio/mpeg", upsert: true });
  if (upErr) bail(`upload ${dest}: ${upErr.message}`);
  console.log(`    → ${dest} (${buf.byteLength} B)`);
  return dest;
}

async function runC3(participant, stim) {
  const dest = `c3_fish_tts/${participant.id}/${stim.stimulus_code}.mp3`;
  if (args.dryRun) {
    console.log(`  ${stim.stimulus_code} C3: would TTS → ${dest}`);
    return dest;
  }
  console.log(`  ${stim.stimulus_code} C3: Fish TTS …`);
  const buf = await fishTTS(stim.sentence, participant.fish_voice_id);
  const { error: upErr } = await supa.storage
    .from("research_stimuli")
    .upload(dest, buf, { contentType: "audio/mpeg", upsert: true });
  if (upErr) bail(`upload ${dest}: ${upErr.message}`);
  console.log(`    → ${dest} (${buf.byteLength} B)`);
  return dest;
}

// ──────────────────────────────────────────────────────────────────────
// Utilities
// ──────────────────────────────────────────────────────────────────────

function parseArgs(argv) {
  const out = { onlyC1: false, participant: null, skipC2: false, skipC3: false, dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--only-c1") out.onlyC1 = true;
    else if (a === "--skip-c2") out.skipC2 = true;
    else if (a === "--skip-c3") out.skipC3 = true;
    else if (a === "--dry-run") out.dryRun = true;
    else if (a === "--participant") out.participant = argv[++i];
    else { console.error(`unknown arg: ${a}`); process.exit(2); }
  }
  return out;
}

function must(name) {
  const v = process.env[name];
  if (!v) bail(`env ${name} is not set`);
  return v;
}

function bail(msg) {
  console.error(`✗ ${msg}`);
  process.exit(1);
}
