// Download all recordings + model stimuli for one participant and write a
// manifest that the existing evaluate_ppg_dtw.py script can consume.
//
// Each (stimulus, test_stage) participant recording is paired with the
// model voice that the participant's Latin-square assignment trained
// against for that stimulus. Pre-test and post-test recordings are
// compared to the *same* model voice so Δ = post − pre is a within-
// participant within-condition measure of shadowing gain.
//
// Run:
//   node --env-file=.env.local scripts/analyze-participant.mjs --code REALTEST01
//
// Flags:
//   --code <external_code>    required
//   --out <dir>               base directory for downloads + manifest (default: ./analysis/<code>)
//   --skip-download           re-use existing files in --out
//
// Produces:
//   <out>/recordings/<stage>/<stimulus_code>.webm  (or .mp3)
//   <out>/models/<stimulus_code>.mp3                 (per-participant assigned model)
//   <out>/models_native/<stimulus_code>.mp3          (C1 shared native reference)
//   <out>/manifest_primary.json     all conditions vs C1 native — for H1 testing
//   <out>/manifest_supplementary.json each condition vs its assigned model — Yamanaka-style
//   <out>/manifest.json              alias of manifest_primary (default for evaluate_ppg_dtw.py)
//   <out>/summary.json               quick inventory + stats
//
// Why two manifests?
//   The paper's H1 (C3 > C2 > C1 in learning gain) implicitly defines the
//   target as native English pronunciation; the cloned/accented voices are
//   teaching scaffolds, not measurement targets. So the *primary* analysis
//   compares all conditions to the same C1 native reference (manifest_primary).
//   The supplementary analysis compares each condition to the voice the
//   participant actually shadowed (Yamanaka 2025-style "voice imitation
//   efficacy" measure).
//
// Next step (after this script):
//   python ../scripts/evaluate_ppg_dtw.py batch \
//      --manifest <out>/manifest.json \
//      --output   <out>/ppg_dtw_results.csv
//
// Then aggregate with:
//   python -c "
//      import pandas as pd
//      r = pd.read_csv('<out>/ppg_dtw_results.csv')
//      print(r.groupby(['condition','test_stage']).distance.mean())
//      pre  = r[r.test_stage=='pre'].set_index('stimulus').distance
//      post = r[r.test_stage=='post'].set_index('stimulus').distance
//      gain = pre - post
//      print(gain.groupby(r.set_index('stimulus').condition).mean())
//   "

import { createClient } from "@supabase/supabase-js";
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";

const args = parseArgs(process.argv.slice(2));
if (!args.code) bail("--code is required");

const env = {
  url: must("NEXT_PUBLIC_SUPABASE_URL"),
  key: must("SUPABASE_SERVICE_ROLE_KEY"),
};

const supa = createClient(env.url, env.key, {
  auth: { autoRefreshToken: false, persistSession: false },
});

const outDir = args.out || `./analysis/${args.code}`;
const recDir = join(outDir, "recordings");
const modelDir = join(outDir, "models");
const nativeDir = join(outDir, "models_native");
await mkdir(join(recDir, "pre"), { recursive: true });
await mkdir(join(recDir, "train"), { recursive: true });
await mkdir(join(recDir, "post"), { recursive: true });
await mkdir(modelDir, { recursive: true });
await mkdir(nativeDir, { recursive: true });

// 1. Resolve participant
const { data: participants, error: pErr } = await supa
  .from("research_participants")
  .select("id, external_code, latin_square_group, fish_voice_id, elevenlabs_voice_id, status")
  .eq("external_code", args.code);
if (pErr || !participants?.length) bail(`participant ${args.code} not found`);
const participant = participants[0];
console.log(`Participant: ${args.code}  id=${participant.id.slice(0, 8)}…  group=${participant.latin_square_group}  status=${participant.status}`);

// 2. Resolve assignments (per (stimulus, stage) → condition + model audio path)
const { data: assignments } = await supa
  .from("research_assignments")
  .select(
    "stimulus_id, condition, test_stage, audio_storage_path, research_stimuli(stimulus_code, sentence, c1_storage_path)",
  )
  .eq("participant_id", participant.id);
if (!assignments?.length) bail("no assignments for participant");

// Build per-stimulus maps:
//   conditionByStimulus[code]   = the participant's assigned condition (1/2/3)
//   assignedModelByStimulus[code] = the model voice participant actually shadowed
//                                   (for the supplementary "vs assigned model" measure)
//   nativeModelByStimulus[code]   = the canonical C1 native reference
//                                   (for the primary "vs native target" measure)
const conditionByStimulus = new Map();
const assignedModelByStimulus = new Map();
const nativeModelByStimulus = new Map();
for (const a of assignments) {
  if (a.test_stage !== "train") continue;
  const code = a.research_stimuli.stimulus_code;
  conditionByStimulus.set(code, a.condition);
  // Native reference: same for all conditions — the shared C1 native track
  nativeModelByStimulus.set(code, a.research_stimuli.c1_storage_path);
  // Assigned model: what the participant shadowed
  //   C1 → c1_native (same as native ref)
  //   C2 → c2_eleven_s2s/<participant_id>/<code>.mp3
  //   C3 → c3_fish_tts/<participant_id>/<code>.mp3
  const assignedPath =
    a.condition === 1
      ? a.research_stimuli.c1_storage_path
      : a.audio_storage_path;
  assignedModelByStimulus.set(code, assignedPath);
}

// 3. Pull all recordings (Storage paths) for participant
const { data: recordings } = await supa
  .from("research_recordings")
  .select("storage_path, duration_sec, mime_type")
  .eq("participant_id", participant.id);
console.log(`Recordings in DB: ${recordings?.length ?? 0}`);

// 4. Download each (skipping if already present)
async function download(bucket, remotePath, localPath) {
  if (args.skipDownload && existsSync(localPath)) return;
  const { data, error } = await supa.storage.from(bucket).download(remotePath);
  if (error) throw new Error(`download ${remotePath}: ${error.message}`);
  const buf = Buffer.from(await data.arrayBuffer());
  await writeFile(localPath, buf);
}

const manifestPrimary = [];
const manifestSupp = [];

// 4a. Download assigned-condition model audio (per-participant)
console.log("\nDownloading assigned model audio (per-participant)…");
for (const [code, modelPath] of assignedModelByStimulus) {
  if (!modelPath) {
    console.log(`  ✗ ${code}: no assigned model path`);
    continue;
  }
  const local = join(modelDir, `${code}.mp3`);
  await download("research_stimuli", modelPath, local);
  console.log(`  ✓ ${code}  assigned ← ${modelPath}`);
}

// 4a-2. Download native reference (shared across all conditions)
console.log("\nDownloading native reference audio…");
for (const [code, nativePath] of nativeModelByStimulus) {
  if (!nativePath) {
    console.log(`  ✗ ${code}: no native reference path`);
    continue;
  }
  const local = join(nativeDir, `${code}.mp3`);
  await download("research_stimuli", nativePath, local);
}
console.log(`  ✓ ${nativeModelByStimulus.size} native reference clips downloaded`);

// 4b. Download recordings + populate manifest
console.log("\nDownloading participant recordings…");
for (const r of recordings ?? []) {
  const parts = r.storage_path.split("/"); // <uid>/<stage>/<file>
  if (parts.length < 3) continue;
  const stage = parts[1];
  const fname = parts[parts.length - 1];
  if (stage === "calibration") continue; // not part of PPG-DTW analysis

  // Extract stimulus_code from filename (e.g. alpha1.webm, alpha1_t01.webm)
  const match = fname.match(/^(alpha\d|beta\d|gamma\d)/);
  if (!match) continue;
  const code = match[1];

  const ext = fname.split(".").pop() || "webm";
  const localUser = join(recDir, stage, `${code}.${ext}`);
  await download("research_recordings", r.storage_path, localUser);

  const condition = conditionByStimulus.get(code);
  const localAssigned = join(modelDir, `${code}.mp3`);
  const localNative = join(nativeDir, `${code}.mp3`);

  if (!existsSync(localNative)) {
    console.log(`  ⚠ ${stage}/${code}: native ref missing, skipping`);
    continue;
  }

  const baseRow = {
    participant: args.code,
    participant_id: participant.id,
    stimulus: code,
    condition: `cond${condition}`,
    test_stage: stage,
    user_wav: localUser,
  };
  // Primary: all conditions vs native reference (for H1 test)
  manifestPrimary.push({ ...baseRow, model_wav: localNative, measure: "vs_native" });
  // Supplementary: each condition vs its assigned model (Yamanaka-style)
  if (existsSync(localAssigned)) {
    manifestSupp.push({ ...baseRow, model_wav: localAssigned, measure: "vs_assigned" });
  }
  console.log(`  ✓ ${stage}/${code}  (cond=${condition})`);
}

// 5. Write both manifests + summary
await writeFile(join(outDir, "manifest_primary.json"), JSON.stringify(manifestPrimary, null, 2));
await writeFile(join(outDir, "manifest_supplementary.json"), JSON.stringify(manifestSupp, null, 2));
// Default alias = primary
await writeFile(join(outDir, "manifest.json"), JSON.stringify(manifestPrimary, null, 2));

const summary = {
  participant: args.code,
  participant_id: participant.id,
  latin_square_group: participant.latin_square_group,
  status: participant.status,
  counts: {
    manifest_primary_entries: manifestPrimary.length,
    manifest_supplementary_entries: manifestSupp.length,
    by_stage: groupCount(manifestPrimary, "test_stage"),
    by_condition: groupCount(manifestPrimary, "condition"),
  },
  next_steps: [
    `# PRIMARY (H1 test, vs native): all conditions compared to Friends original`,
    `python ../scripts/evaluate_ppg_dtw.py batch --manifest ${join(outDir, "manifest_primary.json")} --output ${join(outDir, "ppg_dtw_primary.csv")}`,
    ``,
    `# SUPPLEMENTARY (Yamanaka-style, vs assigned model): each condition vs voice shadowed`,
    `python ../scripts/evaluate_ppg_dtw.py batch --manifest ${join(outDir, "manifest_supplementary.json")} --output ${join(outDir, "ppg_dtw_supplementary.csv")}`,
  ],
};
await writeFile(join(outDir, "summary.json"), JSON.stringify(summary, null, 2));

console.log(`\n=== Done ===`);
console.log(`  Primary manifest:        ${join(outDir, "manifest_primary.json")} (${manifestPrimary.length} entries, vs native)`);
console.log(`  Supplementary manifest:  ${join(outDir, "manifest_supplementary.json")} (${manifestSupp.length} entries, vs assigned model)`);
console.log(`  By stage:     ${JSON.stringify(summary.counts.by_stage)}`);
console.log(`  By condition: ${JSON.stringify(summary.counts.by_condition)}`);
console.log(`\nNext step — run both PPG-DTW analyses:`);
for (const line of summary.next_steps) console.log(`  ${line}`);

function groupCount(arr, key) {
  const c = {};
  for (const x of arr) c[x[key]] = (c[x[key]] ?? 0) + 1;
  return c;
}

function parseArgs(argv) {
  const out = { code: null, out: null, skipDownload: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--code") out.code = argv[++i];
    else if (a === "--out") out.out = argv[++i];
    else if (a === "--skip-download") out.skipDownload = true;
    else if (a === "--help") {
      console.log("Usage: node --env-file=.env.local scripts/analyze-participant.mjs --code <external_code> [--out <dir>] [--skip-download]");
      process.exit(0);
    } else { console.error(`unknown arg: ${a}`); process.exit(2); }
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
