"""Quantify high-confidence role mislabels vs vC1_040 (NO models, NO writes).

MEDIUM (execution-anchored, strict):
  speaker reports >=2 EXECUTED players' alignment, ALL reported targets are
  executed (no living-player reports), and vC1_040 did NOT label them Medium.
  -> strong real-Medium signal the solver missed.

SEER (looser, flagged noisy):
  speaker has >=2 divination results, claims Seer, and is NOT the obvious
  contradicted one in a multi-CO. Listed separately for manual spot-check.

For each Medium fix we also report who vC1_040 currently calls Medium in that
game (that slot must be reassigned -> default Villager).
"""
import json, os
from pathlib import Path
import pandas as pd

EV = Path("artifacts/evidence")
RES = Path("artifacts/llm_results")

base = pd.read_csv("artifacts/submissions/vC1_040_private.csv", dtype={"index": str})
base["index"] = base["index"].str.zfill(2)
role_of, game_rows = {}, {}
for _, r in base.iterrows():
    g = f"private_{r['index']}"
    role_of[(g, r["character"])] = r["role"]
    game_rows.setdefault(g, []).append((r["character"], r["role"]))

med_fixes, seer_cands = [], []
for idx in sorted(base["index"].unique()):
    g = f"private_{idx}"
    ev = json.loads((EV / f"{g}.json").read_text())["evidence"]
    execs = set(ev.get("executions", []))
    nd = set(ev.get("night_deaths", []))
    rp = RES / f"{g}.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    mb = res.get("medium_by", {}) or {}
    db = res.get("divination_by", {}) or {}
    co_s = set(ev.get("co_seer", []))

    cur_mediums = [c for c, ro in game_rows[g] if ro == "Medium"]

    for spk, rep in mb.items():
        tgts = set(rep.get("wolf", [])) | set(rep.get("human", []))
        if not tgts:
            continue
        ex_hits = tgts & execs
        living = tgts - execs - nd
        if len(ex_hits) >= 2 and len(living) == 0:
            cur = role_of.get((g, spk), "?")
            if cur != "Medium":
                med_fixes.append({
                    "game": g, "speaker": spk, "cur_role": cur,
                    "exec_reported": sorted(ex_hits),
                    "displaced_medium": cur_mediums,
                })

    for spk, rep in db.items():
        n = len(set(rep.get("wolf", [])) | set(rep.get("human", [])))
        if n >= 2 and spk in co_s:
            cur = role_of.get((g, spk), "?")
            seer_cands.append({
                "game": g, "speaker": spk, "cur_role": cur, "n_div": n,
                "multi_co": len(co_s) > 1, "co_seer": sorted(co_s),
            })

print("=== STRICT MEDIUM mislabels (>=2 executed reports, 0 living, not Medium in vC1) ===")
for f in med_fixes:
    print(f"{f['game']}: {f['speaker']:<24} vC1={f['cur_role']:<9} "
          f"exec-reported={f['exec_reported']}  displaced-Medium={f['displaced_medium']}")
print(f"\nTOTAL strict Medium fixes: {len(med_fixes)}")

print("\n=== SEER candidates (>=2 div + claims-seer) — NOISY, spot-check ===")
for f in seer_cands:
    flag = "  [MULTI-CO]" if f["multi_co"] else ""
    print(f"{f['game']}: {f['speaker']:<24} vC1={f['cur_role']:<9} "
          f"n_div={f['n_div']}{flag}  co_seer={f['co_seer']}")
print(f"\nTOTAL seer candidates: {len(seer_cands)} "
      f"(non-multi-CO: {sum(1 for f in seer_cands if not f['multi_co'])})")
