"""Private-case audit sheet generator (NO models).

For each private game, emit a compact, scannable block:
  - vC1_040 top-6 wolf ranking + assigned role
  - role counts, executions (ordered), night deaths
  - CO claims, Medium results, Seer results (from evidence + llm_results)
  - AUTO-FLAGS per top-6 suspect (deterministic high-leverage hints)
  - a few relevant quotes per top-6 suspect (self-claim / accusations)

Outputs:
  private_audit_sheet.md   (human review)
  private_audit.json       (machine readable, consumed by apply_patch.py)

Annotation vocabulary (write in the sheet next to a suspect):
  KEEP | BOOST_WOLF_SMALL | BOOST_WOLF_BIG | DEMOTE_WOLF_SMALL |
  DEMOTE_WOLF_BIG | SET_MIN_080 | SET_MAX_015 | SET_MAX_005 |
  ROLE_MEDIUM | ROLE_SEER | ROLE_HUNTER | ROLE_MADMAN |
  ROLE_WEREWOLF | ROLE_VILLAGER
"""
from __future__ import annotations

import json, re
from pathlib import Path
import pandas as pd

from parsers.posts import parse_posts
from parsers.names import build_speaker_map

DATA = Path("/home/htiintern2502/AI2/data/Werewolf_Prediction_Dataset")
EV = Path("artifacts/evidence")
RES = Path("artifacts/llm_results")
BASE = Path("artifacts/submissions/vC1_040_private.csv")

CLAIM_RE = re.compile(
    r"\[\s*(seer|medium|hunter)\s*\]|\bco\s+(seer|medium|hunter)\b"
    r"|i[' ]?a?m\s+(?:the\s+|a\s+)?(seer|medium|hunter)\b"
    r"|my\s+(?:divination|seer|medium)\s+result|i\s+(?:checked|inspected|divined)\b",
    re.IGNORECASE)
ACCU_RE = re.compile(
    r"\b(wolf|werewolf|lying|liar|fake|suspicious|sus|black|evil|scum)\b",
    re.IGNORECASE)


def alias(n):
    return [t.lower() for t in n.replace("_", " ").split() if len(t) >= 3]


def med_reports(ev, res):
    """Merge evidence medium + llm_results medium_by -> {speaker:{wolf:[],human:[]}}"""
    out = {}
    for sp, tg in (ev.get("medium_blacks", {}) or {}).items():
        out.setdefault(sp, {"wolf": [], "human": []})["wolf"] += list(tg)
    for sp, tg in (ev.get("medium_whites", {}) or {}).items():
        out.setdefault(sp, {"wolf": [], "human": []})["human"] += list(tg)
    for sp, d in (res.get("medium_by", {}) or {}).items():
        s = out.setdefault(sp, {"wolf": [], "human": []})
        s["wolf"] += list(d.get("wolf", [])); s["human"] += list(d.get("human", []))
    return {k: {"wolf": sorted(set(v["wolf"])), "human": sorted(set(v["human"]))}
            for k, v in out.items()}


def seer_reports(ev, res):
    out = {}
    for sp, tg in (ev.get("seer_blacks", {}) or {}).items():
        out.setdefault(sp, {"wolf": [], "human": []})["wolf"] += list(tg)
    for sp, tg in (ev.get("seer_whites", {}) or {}).items():
        out.setdefault(sp, {"wolf": [], "human": []})["human"] += list(tg)
    for sp, d in (res.get("divination_by", {}) or {}).items():
        s = out.setdefault(sp, {"wolf": [], "human": []})
        s["wolf"] += list(d.get("wolf", [])); s["human"] += list(d.get("human", []))
    return {k: {"wolf": sorted(set(v["wolf"])), "human": sorted(set(v["human"]))}
            for k, v in out.items()}


def quotes_for(name, posts, spk, k=3):
    """Return up to k short quotes: self role-claims first, then accusations against."""
    toks = alias(name)
    self_q, accu_q = [], []
    for p in posts:
        s = spk.get(p.speaker, p.speaker)
        txt = re.sub(r"\s+", " ", p.text).strip()
        if s == name and CLAIM_RE.search(txt) and len(self_q) < k:
            self_q.append(f"(self d{p.day}) {txt[:170]}")
        elif s != name and ACCU_RE.search(txt):
            low = " " + txt.lower() + " "
            if any(f" {t} " in low or f" {t}." in low or f" {t}," in low
                   for t in toks) and len(accu_q) < k:
                accu_q.append(f"(by {s} d{p.day}) {txt[:170]}")
    return self_q + accu_q[:max(0, k - len(self_q))]


def main():
    base = pd.read_csv(BASE, dtype={"index": str})
    base["index"] = base["index"].str.zfill(2)
    md = ["# Private Case Audit Sheet (base = vC1_040, Kaggle 0.34548)\n",
          "Annotate each flagged suspect with one action token. "
          "Blank = KEEP.\n",
          "Tokens: BOOST_WOLF_SMALL(+.10) BOOST_WOLF_BIG(+.25) "
          "DEMOTE_WOLF_SMALL(-.10) DEMOTE_WOLF_BIG(-.25) SET_MIN_080 "
          "SET_MAX_015 SET_MAX_005 ROLE_<X>\n"]
    machine = {}

    for idx, sub in base.groupby("index", sort=True):
        ev = json.loads((EV / f"private_{idx}.json").read_text())["evidence"]
        rp = RES / f"private_{idx}.json"
        res = json.loads(rp.read_text()) if rp.exists() else {}
        players = sub["character"].tolist()
        log = (DATA / "private" / f"{idx}.txt").read_text(
            encoding="utf-8", errors="replace")
        posts = parse_posts(log, players)
        spk = build_speaker_map(players, {p.speaker for p in posts})

        execs = ev.get("executions", [])
        nd = set(ev.get("night_deaths", []))
        co_s = set(ev.get("co_seer", []))
        co_m = set(ev.get("co_medium", []))
        co_h = set(ev.get("co_hunter", []))
        nsm = set(ev.get("not_seer_medium", []))
        mrep = med_reports(ev, res)
        srep = seer_reports(ev, res)
        med_wolf = {t for d in mrep.values() for t in d["wolf"]}
        med_human = {t for d in mrep.values() for t in d["human"]}
        seer_wolf = {t for d in srep.values() for t in d["wolf"]}

        sub = sub.sort_values("wolf_score", ascending=False).reset_index(drop=True)
        rc = sub["role"].value_counts().to_dict()
        top = sub.head(6)

        md.append(f"\n---\n\n## private_{idx}\n")
        md.append("**Roles (vC1_040):** " + ", ".join(
            f"{r}:{rc.get(r,0)}" for r in
            ["Villager", "Werewolf", "Seer", "Medium", "Hunter", "Madman"]))
        md.append(f"**Executions (in order):** {execs or '—'}")
        md.append(f"**Night deaths:** {sorted(nd) or '—'}")
        md.append(f"**CO:** seer={sorted(co_s)} medium={sorted(co_m)} "
                  f"hunter={sorted(co_h)} not-seer/med={sorted(nsm)}")
        md.append("**Medium results:** " + (
            "; ".join(f"{s}→W{d['wolf']}/H{d['human']}"
                      for s, d in mrep.items() if d['wolf'] or d['human'])
            or "—"))
        md.append("**Seer results:** " + (
            "; ".join(f"{s}→W{d['wolf']}/H{d['human']}"
                      for s, d in srep.items() if d['wolf'] or d['human'])
            or "—"))

        md.append("\n**Top-6 wolf ranking:**")
        g_machine = []
        for i, row in top.iterrows():
            nm = row["character"]; ws = row["wolf_score"]; rl = row["role"]
            flags = []
            if nm in nd:
                flags.append("⚠NIGHT-DEATH→DEMOTE_BIG/SET_MAX_005")
            if nm in med_wolf and nm in set(execs):
                flags.append("🐺MEDIUM-CONFIRMED-WOLF→SET_MIN_080")
            if nm in med_human and nm in set(execs):
                flags.append("👤MEDIUM-CONFIRMED-HUMAN→SET_MAX_015")
            if nm in seer_wolf:
                flags.append("SEER-BLACK")
            if nm in co_s:
                flags.append("claims-SEER")
            if nm in co_m:
                flags.append("claims-MEDIUM")
            if nm in co_h:
                flags.append("claims-HUNTER")
            if nm in nsm:
                flags.append("disclaimed-seer/med")
            fl = ("  " + " ".join(flags)) if flags else ""
            md.append(f" {i+1}. **{nm}**  ws={ws:.3f}  role={rl}{fl}")
            md.append(f"     ACTION: ____")
            for q in quotes_for(nm, posts, spk):
                md.append(f"       - {q}")
            g_machine.append({"rank": i + 1, "character": nm,
                              "ws": round(float(ws), 4), "role": rl,
                              "flags": flags})
        # full role list (compact) for role-change reference
        md.append("\n**Full assignment:** " + " | ".join(
            f"{r['character']}={r['role']}({r['wolf_score']:.2f})"
            for _, r in sub.iterrows()))
        machine[f"private_{idx}"] = {
            "executions": execs, "night_deaths": sorted(nd),
            "co_seer": sorted(co_s), "co_medium": sorted(co_m),
            "co_hunter": sorted(co_h), "med_wolf": sorted(med_wolf),
            "med_human": sorted(med_human), "seer_wolf": sorted(seer_wolf),
            "top6": g_machine}

    Path("private_audit_sheet.md").write_text("\n".join(md), encoding="utf-8")
    Path("private_audit.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2))
    print(f"Wrote private_audit_sheet.md ({len(md)} lines) + private_audit.json "
          f"({len(machine)} games)")


if __name__ == "__main__":
    main()
