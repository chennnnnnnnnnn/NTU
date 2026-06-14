// Seed the audio-survey clip pool (survey_clips) from public/clips/<student>/.
//
// Run:
//   node --env-file=.env.local scripts/seed-clips.mjs
//   (use Node >= 22 — supabase-js needs native WebSocket)
//
// Each clip is one personalised synthesis: <student_id>/<set><n>_<condition>.wav
//   condition: c2  = ElevenLabs self+native accent
//              c3a = Fish self+Chinese accent
//              c3b = ElevenLabs self+Chinese accent
//
// Idempotent: upserts on clip_code, and removes any legacy native clips
// (student_id IS NULL) left over from the earlier 18-clip pool.

import { createClient } from "@supabase/supabase-js";
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!SUPABASE_URL || !SERVICE_ROLE) {
  console.error("Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env.");
  process.exit(1);
}

// sentence_code → spoken sentence (the 18 Friends lines)
const SENTENCES = {
  alpha1: "I can't believe this is happening.",
  alpha2: "I think I shouldn't look directly at them.",
  alpha3: "Half the guys out there have makeup on.",
  alpha4: "Did the girl ever let you ride it?",
  alpha5: "I never had a bike of my own.",
  alpha6: "Hey wait a minute, this one isn't dirty.",
  beta1: "There's nothing wrong with speaking correctly.",
  beta2: "Maybe we can put it in the guest bedroom.",
  beta3: "Wow, I really get crabby when I cook.",
  beta4: "What kind of discount do we get?",
  beta5: "I think it's gonna be really hard.",
  beta6: "I've had a really good time tonight.",
  gamma1: "His first big kid's bike, this is so exciting!",
  gamma2: "You really think that is what he meant?",
  gamma3: "What is the matter with your hand?",
  gamma4: "Of course it's true and it hurts so bad.",
  gamma5: "There's a side of steamed vegetables.",
  gamma6: "Are you going to eat that bread?",
};

const __dirname = dirname(fileURLToPath(import.meta.url));
const clipsRoot = join(__dirname, "..", "public", "clips");

const FNAME_RE = /^(alpha|beta|gamma)([1-6])_(c2|c3a|c3b)\.wav$/;

const rows = [];
for (const student of readdirSync(clipsRoot)) {
  const dir = join(clipsRoot, student);
  let files;
  try {
    files = readdirSync(dir);
  } catch {
    continue; // not a directory
  }
  for (const file of files) {
    const m = FNAME_RE.exec(file);
    if (!m) continue;
    const [, set, num, condition] = m;
    const sentenceCode = `${set}${num}`;
    rows.push({
      clip_code: `${student}_${sentenceCode}_${condition}`,
      sentence: SENTENCES[sentenceCode] ?? null,
      storage_path: `/clips/${student}/${file}`,
      student_id: student,
      stimulus_set: set,
      sentence_code: sentenceCode,
      condition,
    });
  }
}

if (rows.length === 0) {
  console.error(`No clips found under ${clipsRoot}/<student>/. Copy the wav files first.`);
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE, {
  auth: { autoRefreshToken: false, persistSession: false },
});

const { data, error } = await supabase
  .from("survey_clips")
  .upsert(rows, { onConflict: "clip_code" })
  .select("clip_code");
if (error) {
  console.error("Seed failed:", error);
  process.exit(1);
}

// Remove legacy native clips from the old 18-clip pool (no student_id).
const { error: delErr } = await supabase
  .from("survey_clips")
  .delete()
  .is("student_id", null);
if (delErr) console.warn("Could not remove legacy clips:", delErr.message);

console.log(`Seeded ${data.length} clips across 6 students × 18 sentences × 3 conditions.`);
