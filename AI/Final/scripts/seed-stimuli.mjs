// Seed the 18 Friends sentences into research_stimuli.
//
// Run:
//   node --env-file=.env.local scripts/seed-stimuli.mjs
//
// Idempotent — uses upsert on stimulus_code so re-runs are safe.
//
// Source of truth: ../stimuli/stimuli_18_friends.md (course Part 2 deliverable).
// Each row carries:
//   stimulus_code:        "alpha1" | "beta2" | "gamma6" ...
//   set_label:            "alpha" | "beta" | "gamma"
//   sentence:             verbatim text (display via serif font)
//   source_episode:       "S5:E5" etc
//   syllable_count:       manual count (Merriam-Webster syllabification)
//   target_phoneme_count: total tokens across 5 target categories
//   target_phonemes:      {theta_eth, v, ae_eh, i_ih, n_ng_final}

import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SERVICE_ROLE = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SERVICE_ROLE) {
  console.error("Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env.");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE, {
  auth: { autoRefreshToken: false, persistSession: false },
});

/** @type {const} */
const STIMULI = [
  // ─── Set α ───
  { code: "alpha1", set: "alpha", ep: "S5:E5",  syl:  9, ph: { theta_eth: 1, v: 1, ae_eh: 3, i_ih: 5, n_ng_final: 1 }, sent: "I can't believe this is happening." },
  { code: "alpha2", set: "alpha", ep: "S6:E8",  syl: 11, ph: { theta_eth: 2, v: 0, ae_eh: 2, i_ih: 3, n_ng_final: 0 }, sent: "I think I shouldn't look directly at them." },
  { code: "alpha3", set: "alpha", ep: "S6:E8",  syl:  9, ph: { theta_eth: 2, v: 1, ae_eh: 1, i_ih: 0, n_ng_final: 1 }, sent: "Half the guys out there have makeup on." },
  { code: "alpha4", set: "alpha", ep: "S7:E9",  syl:  9, ph: { theta_eth: 1, v: 1, ae_eh: 2, i_ih: 2, n_ng_final: 0 }, sent: "Did the girl ever let you ride it?" },
  { code: "alpha5", set: "alpha", ep: "S7:E9",  syl:  9, ph: { theta_eth: 0, v: 2, ae_eh: 2, i_ih: 0, n_ng_final: 1 }, sent: "I never had a bike of my own." },
  { code: "alpha6", set: "alpha", ep: "S8:E12", syl: 11, ph: { theta_eth: 1, v: 0, ae_eh: 0, i_ih: 4, n_ng_final: 1 }, sent: "Hey wait a minute, this one isn't dirty." },

  // ─── Set β ───
  { code: "beta1",  set: "beta",  ep: "S1:E3",  syl: 10, ph: { theta_eth: 3, v: 0, ae_eh: 1, i_ih: 2, n_ng_final: 3 }, sent: "There's nothing wrong with speaking correctly." },
  { code: "beta2",  set: "beta",  ep: "S8:E12", syl: 11, ph: { theta_eth: 1, v: 0, ae_eh: 3, i_ih: 3, n_ng_final: 2 }, sent: "Maybe we can put it in the guest bedroom." },
  { code: "beta3",  set: "beta",  ep: "S4:E18", syl: 10, ph: { theta_eth: 0, v: 0, ae_eh: 3, i_ih: 2, n_ng_final: 1 }, sent: "Wow, I really get crabby when I cook." },
  { code: "beta4",  set: "beta",  ep: "S3:E12", syl:  8, ph: { theta_eth: 0, v: 1, ae_eh: 1, i_ih: 2, n_ng_final: 2 }, sent: "What kind of discount do we get?" },
  { code: "beta5",  set: "beta",  ep: "S1:E3",  syl:  9, ph: { theta_eth: 1, v: 0, ae_eh: 0, i_ih: 4, n_ng_final: 0 }, sent: "I think it's gonna be really hard." },
  { code: "beta6",  set: "beta",  ep: "S6:E8",  syl:  9, ph: { theta_eth: 0, v: 1, ae_eh: 1, i_ih: 2, n_ng_final: 0 }, sent: "I've had a really good time tonight." },

  // ─── Set γ ───
  { code: "gamma1", set: "gamma", ep: "S7:E9",  syl: 11, ph: { theta_eth: 1, v: 0, ae_eh: 0, i_ih: 7, n_ng_final: 1 }, sent: "His first big kid's bike, this is so exciting!" },
  { code: "gamma2", set: "gamma", ep: "S3:E12", syl:  9, ph: { theta_eth: 2, v: 0, ae_eh: 2, i_ih: 4, n_ng_final: 0 }, sent: "You really think that is what he meant?" },
  { code: "gamma3", set: "gamma", ep: "S8:E12", syl:  8, ph: { theta_eth: 2, v: 0, ae_eh: 2, i_ih: 2, n_ng_final: 1 }, sent: "What is the matter with your hand?" },
  { code: "gamma4", set: "gamma", ep: "S6:E8",  syl:  9, ph: { theta_eth: 0, v: 1, ae_eh: 2, i_ih: 2, n_ng_final: 1 }, sent: "Of course it's true and it hurts so bad." },
  { code: "gamma5", set: "gamma", ep: "S8:E12", syl:  9, ph: { theta_eth: 1, v: 2, ae_eh: 1, i_ih: 2, n_ng_final: 0 }, sent: "There's a side of steamed vegetables." },
  { code: "gamma6", set: "gamma", ep: "S6:E8",  syl:  8, ph: { theta_eth: 1, v: 0, ae_eh: 2, i_ih: 1, n_ng_final: 1 }, sent: "Are you going to eat that bread?" },
];

const rows = STIMULI.map((s) => ({
  stimulus_code: s.code,
  set_label: s.set,
  sentence: s.sent,
  source_episode: s.ep,
  syllable_count: s.syl,
  target_phoneme_count: Object.values(s.ph).reduce((a, b) => a + b, 0),
  target_phonemes: s.ph,
}));

const { data, error } = await supabase
  .from("research_stimuli")
  .upsert(rows, { onConflict: "stimulus_code" })
  .select("stimulus_code, set_label, syllable_count, target_phoneme_count");

if (error) {
  console.error("Seed failed:", error);
  process.exit(1);
}

console.log(`Seeded ${data.length} stimuli:`);
const by = { alpha: [], beta: [], gamma: [] };
for (const r of data) by[r.set_label].push(r);
for (const set of ["alpha", "beta", "gamma"]) {
  console.log(`  ${set}: ${by[set].length} rows (codes: ${by[set].map((r) => r.stimulus_code).join(", ")})`);
}

const totalSyl = data.reduce((a, r) => a + r.syllable_count, 0);
const totalPh = data.reduce((a, r) => a + r.target_phoneme_count, 0);
console.log(`  Σ syllables: ${totalSyl}, Σ target phonemes: ${totalPh}`);
