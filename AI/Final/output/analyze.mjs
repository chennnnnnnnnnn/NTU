// Analyse the clip-survey A/B comparison results from the exported CSVs.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const dir = dirname(fileURLToPath(import.meta.url));

// --- tiny CSV parser (handles quoted fields with commas) ---
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
      else field += c;
    } else {
      if (c === '"') q = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else if (c === "\r") { /* skip */ }
      else field += c;
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const head = rows.shift();
  return rows.filter(r => r.length > 1).map(r => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}

const P = parseCSV(readFileSync(join(dir, "survey_participants_rows.csv"), "utf8"));
const I = parseCSV(readFileSync(join(dir, "survey_items_rows.csv"), "utf8"));

const pById = new Map(P.map(p => [p.id, p]));

// binomial two-sided sign test p-value (exact)
function logFact(n){let s=0;for(let i=2;i<=n;i++)s+=Math.log(i);return s;}
function binom(n,k){return Math.exp(logFact(n)-logFact(k)-logFact(n-k))*Math.pow(0.5,n);}
function signTest(wins,losses){
  const n=wins+losses; if(n===0)return 1; const k=Math.min(wins,losses);
  let p=0; for(let i=0;i<=k;i++)p+=binom(n,i); return Math.min(1,2*p);
}

console.log("══════════════ 受試者概況 ══════════════");
console.log("總人數:", P.length);
const tally = (key) => P.reduce((m,p)=>{const v=p[key]||"(空)";m[v]=(m[v]||0)+1;return m;},{});
console.log("狀態:", JSON.stringify(tally("status")));
console.log("注意力檢查:", JSON.stringify(tally("attention_passed")));
console.log("年齡:", JSON.stringify(tally("age_bracket")));
console.log("性別:", JSON.stringify(tally("gender")));
console.log("母語:", JSON.stringify(tally("native_language")));
console.log("英語程度:", JSON.stringify(tally("english_level")));
console.log("戴耳機:", JSON.stringify(tally("used_headphones")));

// attention failers
const failers = P.filter(p => p.attention_passed === "false");
console.log("\n未通過注意力檢查者:", failers.length ? failers.map(p=>p.team_member_name||p.external_code).join(", ") : "(無)");

// valid sample = completed & attention_passed
const validIds = new Set(P.filter(p => p.status==="completed" && p.attention_passed==="true").map(p=>p.id));
console.log("\n有效樣本(完成且通過注意力):", validIds.size, "人");

// comparison items from valid participants, answered
const comps = I.filter(it => it.item_type==="comparison" && it.rating && validIds.has(it.participant_id));
console.log("有效比較作答數:", comps.length);

// For a question, target = the condition we report preference toward.
// rating 1..5 : 1 = strongly A, 5 = strongly B.
// pref toward C: if A==C -> 6-rating ; if B==C -> rating. (>3 prefers C)
function analyse(qtype, target, other, label, hint) {
  const rows = comps.filter(c => c.question_type === qtype);
  let sum=0, wins=0, losses=0, ties=0;
  const dist={1:0,2:0,3:0,4:0,5:0};
  for (const c of rows) {
    const r = Number(c.rating);
    const pref = c.condition_a === target ? 6 - r : r; // pref toward target
    dist[pref]=(dist[pref]||0)+1;
    sum += pref;
    if (pref>3) wins++; else if (pref<3) losses++; else ties++;
  }
  const n = rows.length;
  const mean = n ? (sum/n) : 0;
  const p = signTest(wins, losses);
  console.log(`\n──────── ${label} (${qtype}) ────────`);
  console.log(`配對: ${target} vs ${other}  | N = ${n}`);
  console.log(`平均偏好分 (3=中立, >3 偏 ${target}): ${mean.toFixed(2)}`);
  console.log(`偏 ${target}: ${wins} 票 (${(100*wins/n).toFixed(0)}%) | 偏 ${other}: ${losses} 票 (${(100*losses/n).toFixed(0)}%) | 平手: ${ties}`);
  console.log(`偏好分分布 1→${target}強 ... 5→${target}最強: ${[1,2,3,4,5].map(k=>dist[k]).join(" / ")}`);
  console.log(`sign test p ≈ ${p.toFixed(4)} ${p<0.05?"(顯著)":"(未達顯著)"}`);
  console.log(`解讀: ${hint}`);

  // per-set (alpha/beta/gamma) breakdown
  const setOf = (code) => (code||"").replace(/[0-9]+$/,"");
  console.log(`  依句組細分 (偏 ${target} 的程度):`);
  for (const set of ["alpha","beta","gamma"]) {
    const rs = rows.filter(c => setOf(c.sentence_code) === set);
    if (!rs.length) { console.log(`   ${set}: (無)`); continue; }
    let s2=0,w=0,l=0,t=0;
    for (const c of rs) { const r=Number(c.rating); const pf = c.condition_a===target ? 6-r : r; s2+=pf; if(pf>3)w++;else if(pf<3)l++;else t++; }
    const m=(s2/rs.length);
    console.log(`   ${set}: N=${rs.length}  平均=${m.toFixed(2)}  偏${target} ${w}(${(100*w/rs.length).toFixed(0)}%) / 偏${other} ${l}(${(100*l/rs.length).toFixed(0)}%) / 平手 ${t}  | sign p≈${signTest(w,l).toFixed(3)}`);
  }
}

console.log("\n══════════════ 三題 A/B 分析 (有效樣本) ══════════════");
analyse("q1_accent",  "c2",  "c3b", "Q1 口音 — 哪段更像母語者", "c2=ElevenLabs母語腔, c3b=ElevenLabs中式腔。偏 c2 越多 → 口音操控越成功。");
analyse("q2_human",   "c3a", "c3b", "Q2 真人 — 哪段更像真人",   "c3a=Fish中式腔, c3b=ElevenLabs中式腔。看哪種合成更像真人。");
analyse("q3_natural", "c2",  "c3a", "Q3 自然 — 哪段更自然",     "c2=ElevenLabs母語腔, c3a=Fish中式腔。偏 c2 → 換口音的音色代價聽得出來。");

// per-speaker sanity (how many comparisons per voice)
const bySpk = comps.reduce((m,c)=>{m[c.student_id]=(m[c.student_id]||0)+1;return m;},{});
console.log("\n各 speaker 出現比較數:", JSON.stringify(bySpk));
