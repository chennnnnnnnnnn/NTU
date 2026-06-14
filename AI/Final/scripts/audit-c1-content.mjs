// Audit C1 native audio against the DB sentence field.
//
// For each row in research_stimuli, download c1_storage_path from Supabase
// Storage, run it through ElevenLabs Scribe speech-to-text, normalise both
// strings, and compute a SequenceMatcher similarity ratio. Anything below
// the threshold is flagged so the researcher can either retrim the audio
// or update the sentence field to match what was actually recorded.
//
// Run:
//   node --env-file=.env.local scripts/audit-c1-content.mjs
//
// Flags:
//   --threshold 0.70   minimum similarity ratio considered a match
//   --code alpha1      restrict to a single stimulus_code
//   --json out.json    also write a structured report
//   --quiet            suppress per-clip line, only print summary
//
// Required env:
//   NEXT_PUBLIC_SUPABASE_URL
//   SUPABASE_SERVICE_ROLE_KEY
//   ELEVENLABS_API_KEY

import { createClient } from "@supabase/supabase-js";
import { writeFileSync } from "node:fs";

const args = parseArgs(process.argv.slice(2));
const env = {
  url: must("NEXT_PUBLIC_SUPABASE_URL"),
  key: must("SUPABASE_SERVICE_ROLE_KEY"),
  eleven: must("ELEVENLABS_API_KEY"),
};

const supa = createClient(env.url, env.key, {
  auth: { autoRefreshToken: false, persistSession: false },
});

// ──────────────────────────────────────────────────────────────────────
// 1. Load DB rows
// ──────────────────────────────────────────────────────────────────────

let query = supa
  .from("research_stimuli")
  .select("stimulus_code, sentence, c1_storage_path")
  .order("stimulus_code");
if (args.code) query = query.eq("stimulus_code", args.code);

const { data: rows, error } = await query;
if (error) bail(`load research_stimuli: ${error.message}`);
if (!rows || rows.length === 0) bail("no rows returned");

if (!args.quiet) {
  console.log(`Auditing ${rows.length} stimuli (threshold ${args.threshold}):\n`);
}

// ──────────────────────────────────────────────────────────────────────
// 2. STT each clip, compare to expected
// ──────────────────────────────────────────────────────────────────────

const results = [];
for (const row of rows) {
  const code = row.stimulus_code;
  const path = row.c1_storage_path;
  const expected = row.sentence;
  if (!path) {
    results.push({ code, status: "no_audio", expected, transcribed: null, similarity: 0 });
    if (!args.quiet) console.log(`  ${pad(code, 8)}  no c1_storage_path`);
    continue;
  }

  try {
    const { data: blob, error: dErr } = await supa.storage
      .from("research_stimuli")
      .download(path);
    if (dErr || !blob) throw new Error(`download ${path}: ${dErr?.message ?? "unknown"}`);

    const transcribed = await elevenLabsScribe(blob);
    const similarity = sequenceMatcher(normalise(transcribed), normalise(expected));
    const status = similarity >= args.threshold ? "ok" : "mismatch";

    results.push({ code, status, expected, transcribed, similarity });

    if (!args.quiet) {
      const flag = status === "ok" ? "✓" : "✗ MISMATCH";
      console.log(`  ${pad(code, 8)}  sim=${similarity.toFixed(2)}  ${flag}`);
      if (status === "mismatch") {
        console.log(`             expected:    "${truncate(expected, 80)}"`);
        console.log(`             transcribed: "${truncate(transcribed, 80)}"`);
      }
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    results.push({ code, status: "error", expected, transcribed: null, similarity: 0, error: msg });
    if (!args.quiet) console.log(`  ${pad(code, 8)}  ✗ ${msg.slice(0, 70)}`);
  }
}

// ──────────────────────────────────────────────────────────────────────
// 3. Summary
// ──────────────────────────────────────────────────────────────────────

const summary = {
  total: results.length,
  ok: results.filter((r) => r.status === "ok").length,
  mismatch: results.filter((r) => r.status === "mismatch").length,
  error: results.filter((r) => r.status === "error").length,
  noAudio: results.filter((r) => r.status === "no_audio").length,
};

console.log(`\n=== Summary ===`);
console.log(`  ✓ matches:    ${summary.ok}/${summary.total}`);
if (summary.mismatch > 0) {
  const bad = results.filter((r) => r.status === "mismatch").map((r) => r.code).join(", ");
  console.log(`  ✗ mismatches: ${summary.mismatch}   ${bad}`);
}
if (summary.error > 0) {
  const errs = results.filter((r) => r.status === "error").map((r) => r.code).join(", ");
  console.log(`  ! errors:     ${summary.error}   ${errs}`);
}
if (summary.noAudio > 0) {
  console.log(`  ? no audio:   ${summary.noAudio}`);
}

if (args.json) {
  writeFileSync(args.json, JSON.stringify({ summary, results }, null, 2));
  console.log(`\nReport saved → ${args.json}`);
}

process.exit(summary.mismatch > 0 ? 1 : 0);

// ──────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────

async function elevenLabsScribe(blob) {
  const fd = new FormData();
  fd.append("file", blob, "clip.mp3");
  fd.append("model_id", "scribe_v1");
  fd.append("language_code", "eng");

  const res = await fetch("https://api.elevenlabs.io/v1/speech-to-text", {
    method: "POST",
    headers: { "xi-api-key": env.eleven },
    body: fd,
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Scribe ${res.status}: ${t.slice(0, 150)}`);
  }
  const json = await res.json();
  return (json.text ?? "").trim();
}

function normalise(s) {
  return s
    .toLowerCase()
    .replace(/\([^)]*\)/g, "") // strip "(beep)", "(laughter)"
    .replace(/[^a-z0-9' ]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Python difflib.SequenceMatcher ratio() port. */
function sequenceMatcher(a, b) {
  if (!a && !b) return 1;
  if (!a || !b) return 0;
  const m = matchingBlocks(a, b);
  const matches = m.reduce((sum, [, , size]) => sum + size, 0);
  return (2 * matches) / (a.length + b.length);
}

function matchingBlocks(a, b) {
  const blocks = [];
  function find(alo, ahi, blo, bhi) {
    const counts = new Map();
    let bestI = alo;
    let bestJ = blo;
    let bestSize = 0;
    for (let i = alo; i < ahi; i++) {
      const ch = a[i];
      const matches = [];
      for (let j = blo; j < bhi; j++) {
        if (b[j] === ch) matches.push(j);
      }
      const newCounts = new Map();
      for (const j of matches) {
        const k = (counts.get(j - 1) ?? 0) + 1;
        newCounts.set(j, k);
        if (k > bestSize) {
          bestI = i - k + 1;
          bestJ = j - k + 1;
          bestSize = k;
        }
      }
      counts.clear();
      newCounts.forEach((v, k) => counts.set(k, v));
    }
    if (bestSize === 0) return;
    if (alo < bestI && blo < bestJ) find(alo, bestI, blo, bestJ);
    blocks.push([bestI, bestJ, bestSize]);
    if (bestI + bestSize < ahi && bestJ + bestSize < bhi)
      find(bestI + bestSize, ahi, bestJ + bestSize, bhi);
  }
  find(0, a.length, 0, b.length);
  return blocks;
}

function parseArgs(argv) {
  const out = { threshold: 0.7, code: null, json: null, quiet: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--threshold") out.threshold = parseFloat(argv[++i]);
    else if (a === "--code") out.code = argv[++i];
    else if (a === "--json") out.json = argv[++i];
    else if (a === "--quiet") out.quiet = true;
    else if (a === "--help") {
      console.log(
        "Usage: node --env-file=.env.local scripts/audit-c1-content.mjs [--threshold 0.7] [--code <code>] [--json out.json] [--quiet]",
      );
      process.exit(0);
    } else {
      console.error(`unknown arg: ${a}`);
      process.exit(2);
    }
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

function pad(s, n) {
  return (s ?? "").padEnd(n, " ");
}

function truncate(s, n) {
  if (!s) return "";
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}
