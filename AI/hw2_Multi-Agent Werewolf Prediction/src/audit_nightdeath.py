"""AUDIT ONLY (no writes, no submission): is R1 (night_death -> not Werewolf)
actually holding in manual_role_fingerprint_safe, or did the solver slip
night victims back into Werewolf / high wolf_score?

Checks per private game:
  A. night_death players whose final role == Werewolf   (ironclad error)
  B. night_death players with wolf_score > 0.2           (AP interference)
Also prints the wolf_score distribution of night-death players so we can
see if R1 already crushed them.
"""
import json
from pathlib import Path
import pandas as pd

EV = Path("artifacts/evidence")
SUB = Path("artifacts/submissions/manual_role_fingerprint_safe_private.csv")

df = pd.read_csv(SUB, dtype={"index": str})
df["index"] = df["index"].str.zfill(2)

roleA, wsB, allnd = [], [], []
for idx in sorted(df["index"].unique()):
    g = f"private_{idx}"
    ev = json.loads((EV / f"{g}.json").read_text())["evidence"]
    nd = set(ev.get("night_deaths", []) or [])
    execed = set(ev.get("executions", []) or [])
    sub = df[df["index"] == idx]
    for _, r in sub.iterrows():
        if r["character"] not in nd:
            continue
        also_exec = r["character"] in execed   # ambiguous: died both ways?
        allnd.append((g, r["character"], r["role"], round(r["wolf_score"], 3),
                      also_exec))
        if r["role"] == "Werewolf" and not also_exec:
            roleA.append((g, r["character"], round(r["wolf_score"], 3)))
        if r["wolf_score"] > 0.2 and not also_exec:
            wsB.append((g, r["character"], r["role"], round(r["wolf_score"], 3)))

print(f"Total night-death player rows: {len(allnd)}")
ws_vals = [w for *_, w, _ in allnd]
if ws_vals:
    s = pd.Series(ws_vals)
    print(f"night-death wolf_score: min={s.min():.3f} mean={s.mean():.3f} "
          f"max={s.max():.3f}  (R1 working => should be low)")

print("\n=== A. night-death AND final role == Werewolf (IRONCLAD error) ===")
for g, c, w in roleA:
    print(f"  {g}: {c:<24} role=Werewolf ws={w}")
print(f"TOTAL role-hit: {len(roleA)}")

print("\n=== B. night-death AND wolf_score > 0.2 (AP interference) ===")
for g, c, ro, w in sorted(wsB, key=lambda x: -x[3]):
    print(f"  {g}: {c:<24} role={ro:<9} ws={w}")
print(f"TOTAL ws-hit: {len(wsB)}")

amb = [x for x in allnd if x[4]]
if amb:
    print(f"\n[note] {len(amb)} night-death players also in executions "
          f"(ambiguous death, excluded from fixes):")
    for g, c, ro, w, _ in amb:
        print(f"  {g}: {c} role={ro} ws={w}")
