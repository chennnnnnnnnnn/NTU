"""Boy-Peter fingerprint Medium quantification (NO models, NO writes).

Qualifies iff ALL hold:
  - speaker in co_medium            (explicitly claimed Medium)
  - speaker NOT in co_seer          (no dual claim -> excludes Friedel)
  - speaker has NO divination_by    (pure medium -> excludes Otto)
  - >=1 medium report on an EXECUTED player
  - 0 medium reports on a LIVING player (executed / night-dead only)
  - vC1_040 did NOT already label them Medium

Also reports the displaced vC1 Medium (default replacement -> Villager,
the majority class; flag if displaced looks special).
"""
import json
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

fixes = []
for idx in sorted(base["index"].unique()):
    g = f"private_{idx}"
    ev = json.loads((EV / f"{g}.json").read_text())["evidence"]
    execs = set(ev.get("executions", []))
    nd = set(ev.get("night_deaths", []))
    co_m = set(ev.get("co_medium", []))
    co_s = set(ev.get("co_seer", []))
    rp = RES / f"{g}.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    mb = res.get("medium_by", {}) or {}
    db = res.get("divination_by", {}) or {}
    cur_mediums = [c for c, ro in game_rows[g] if ro == "Medium"]

    for spk in co_m:
        if spk in co_s:
            continue
        if db.get(spk):
            continue
        rep = mb.get(spk)
        if not rep:
            continue
        tgts = set(rep.get("wolf", [])) | set(rep.get("human", []))
        ex_hits = tgts & execs
        living = tgts - execs - nd
        if len(ex_hits) >= 1 and len(living) == 0:
            cur = role_of.get((g, spk), "?")
            if cur != "Medium":
                fixes.append({
                    "game": g, "speaker": spk, "cur": cur,
                    "n_exec": len(ex_hits), "exec": sorted(ex_hits),
                    "wolf": rep.get("wolf", []), "human": rep.get("human", []),
                    "displaced": cur_mediums,
                    "is_bp": (g == "private_12" and spk == "Boy Peter"),
                })

fixes.sort(key=lambda f: (-f["n_exec"], f["game"]))
print("=== CLEAN Medium fixes (Boy-Peter fingerprint) ===")
for f in fixes:
    tag = "  <<< already in role_safe" if f["is_bp"] else ""
    print(f"{f['game']}: {f['speaker']:<22} vC1={f['cur']:<9} "
          f"exec={f['exec']} W{f['wolf']}/H{f['human']} "
          f"displaced={f['displaced']}{tag}")
print(f"\nTOTAL clean: {len(fixes)}  | NEW (excl Boy Peter): "
      f"{sum(1 for f in fixes if not f['is_bp'])}")
