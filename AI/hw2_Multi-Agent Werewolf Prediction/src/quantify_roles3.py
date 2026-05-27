"""Sole/clean Seer & Hunter fingerprints — mirror of the Boy-Peter Medium
rule (+0.0038). Strict per user. NO models, NO writes.

SEER fingerprint (IRONCLAD tier):
  - in co_seer
  - NOT in co_medium  and  NOT in medium_by   (no dual / medium-style)
  - NOT in co_hunter                           (no triple claim)
  - NOT in not_seer_medium                     (never disclaimed)
  - has >=1 divination_by record
  - AND ( single seer claimant  OR  >=2 divinations )
  - vC1_040 did NOT label them Seer
  Chaos guard: if multi-CO seer (>=4 claimants) AND not sole -> 2nd tier.

HUNTER fingerprint (even stricter):
  - co_hunter == exactly 1 person (explicit quote-validated hunter claim)
  - NOT in co_seer / co_medium / not_seer_medium
  - vC1_040 did NOT label them Hunter
  - NOT a top-3 wolf_score suspect in that game (exclude wolf fake-claim)
  - chaos guard: total distinct claimants (seer+med+hunter) <= 5
"""
import json
from pathlib import Path
import pandas as pd

EV = Path("artifacts/evidence")
RES = Path("artifacts/llm_results")

base = pd.read_csv("artifacts/submissions/vC1_040_private.csv", dtype={"index": str})
base["index"] = base["index"].str.zfill(2)
role_of, game_rows, ws_rank = {}, {}, {}
for idx, sub in base.groupby("index"):
    g = f"private_{idx}"
    s = sub.sort_values("wolf_score", ascending=False).reset_index(drop=True)
    for rk, (_, r) in enumerate(s.iterrows()):
        ws_rank[(g, r["character"])] = rk + 1
for _, r in base.iterrows():
    g = f"private_{r['index']}"
    role_of[(g, r["character"])] = r["role"]
    game_rows.setdefault(g, []).append((r["character"], r["role"]))

seer_iron, seer_2nd, hunter_iron = [], [], []
for idx in sorted(base["index"].unique()):
    g = f"private_{idx}"
    ev = json.loads((EV / f"{g}.json").read_text())["evidence"]
    co_s = set(ev.get("co_seer", []) or [])
    co_m = set(ev.get("co_medium", []) or [])
    co_h = set(ev.get("co_hunter", []) or [])
    nsm = set(ev.get("not_seer_medium", []) or [])
    rp = RES / f"{g}.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    db = res.get("divination_by", {}) or {}
    mb = res.get("medium_by", {}) or {}
    cur_seers = [c for c, ro in game_rows[g] if ro == "Seer"]
    cur_hunters = [c for c, ro in game_rows[g] if ro == "Hunter"]
    n_claim = len(co_s | co_m | co_h)

    for s in co_s:
        if s in co_m or s in co_h or s in nsm or mb.get(s):
            continue
        rep = db.get(s) or {}
        ndiv = len(set(rep.get("wolf", [])) | set(rep.get("human", [])))
        if ndiv < 1:
            continue
        cur = role_of.get((g, s), "?")
        if cur == "Seer":
            continue
        sole = len(co_s) == 1
        rec = {"game": g, "spk": s, "cur": cur, "ndiv": ndiv, "div": rep,
               "n_seer_co": len(co_s), "displaced": cur_seers}
        if sole or ndiv >= 2:
            if sole and len(co_s) < 4:
                seer_iron.append(rec)
            elif not sole and len(co_s) >= 4:
                seer_2nd.append(rec)
            elif sole:
                seer_iron.append(rec)
            else:
                seer_2nd.append(rec)

    if len(co_h) == 1:
        h = next(iter(co_h))
        cur = role_of.get((g, h), "?")
        if (h not in co_s and h not in co_m and h not in nsm
                and cur != "Hunter" and ws_rank.get((g, h), 99) > 3
                and n_claim <= 5):
            hunter_iron.append({"game": g, "spk": h, "cur": cur,
                                "ws_rank": ws_rank.get((g, h)),
                                "displaced": cur_hunters})

print("=== SEER fingerprint — IRONCLAD (sole claimant) ===")
for f in seer_iron:
    print(f"{f['game']}: {f['spk']:<22} vC1={f['cur']:<9} ndiv={f['ndiv']} "
          f"co_seer#={f['n_seer_co']} div={f['div']} displaced={f['displaced']}")
print(f"TOTAL ironclad seer: {len(seer_iron)}")

print("\n=== SEER fingerprint — 2ND TIER (>=2 div but multi-CO; likely reject) ===")
for f in seer_2nd:
    print(f"{f['game']}: {f['spk']:<22} vC1={f['cur']:<9} ndiv={f['ndiv']} "
          f"co_seer#={f['n_seer_co']}")
print(f"TOTAL 2nd-tier seer: {len(seer_2nd)}")

print("\n=== HUNTER fingerprint — IRONCLAD (sole co_hunter, no dual, not wolf-top) ===")
for f in hunter_iron:
    print(f"{f['game']}: {f['spk']:<22} vC1={f['cur']:<9} "
          f"ws_rank={f['ws_rank']} displaced={f['displaced']}")
print(f"TOTAL ironclad hunter: {len(hunter_iron)}")
