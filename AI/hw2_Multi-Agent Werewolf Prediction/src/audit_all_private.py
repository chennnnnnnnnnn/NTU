"""FULL per-game structural audit of all private 01-30 (AUDIT ONLY, no writes).

The 'major discovery': each game only SCORES the subset of players listed in
roles.csv, and the solver can collapse that subset into a degenerate role
distribution (e.g. private_08 = all 3 scored players -> Werewolf, one of which
is a night-death = ironclad-impossible).

For every game prints, for each SCORED player:
  role, wolf_score, death-anchor (EXEC / NIGHT / alive)
Flags:
  [IRONCLAD] night-death AND role == Werewolf  (wolves never night-kill wolves)
  [DEGEN]    every scored player has the same role
  [ALLWOLF]  every scored player == Werewolf
Base = manual_nightdeath_fix (current best candidate) so we see remaining
ironclad violations after the Gerd fix.
"""
import json
from pathlib import Path
import pandas as pd

EV = Path("artifacts/evidence")
SUB = Path("artifacts/submissions/manual_nightdeath_fix_private.csv")

df = pd.read_csv(SUB, dtype={"index": str})
df["index"] = df["index"].str.zfill(2)

iron, degen, allwolf = [], [], []
for idx in sorted(df["index"].unique()):
    g = f"private_{idx}"
    ev = json.loads((EV / f"{g}.json").read_text())["evidence"]
    nd = set(ev.get("night_deaths", []) or [])
    ex = set(ev.get("executions", []) or [])
    sub = df[df["index"] == idx].sort_values("wolf_score", ascending=False)
    roles = sub["role"].tolist()
    same = len(set(roles)) == 1
    allw = same and roles[0] == "Werewolf"
    if same:
        degen.append((g, len(sub), roles[0]))
    if allw:
        allwolf.append(g)
    rows = []
    for _, r in sub.iterrows():
        c = r["character"]
        anc = "NIGHT" if c in nd else ("EXEC" if c in ex else "alive")
        rows.append((c, r["role"], round(r["wolf_score"], 3), anc))
        if c in nd and c not in ex and r["role"] == "Werewolf":
            iron.append((g, c, round(r["wolf_score"], 3)))
    tag = ""
    if allw:
        tag = "  <<< ALL-WOLF degenerate"
    elif same:
        tag = f"  <<< all '{roles[0]}'"
    print(f"\n## {g}  (scored {len(sub)}){tag}")
    for c, ro, w, anc in rows:
        mark = "  [IRONCLAD night-death!=Werewolf]" if (anc == "NIGHT" and ro == "Werewolf") else ""
        print(f"   {c:<24} {ro:<9} ws={w:<6} {anc}{mark}")

print("\n" + "=" * 60)
print(f"[IRONCLAD] night-death labelled Werewolf (remaining): {len(iron)}")
for g, c, w in iron:
    print(f"   {g}: {c} ws={w}  -> must be non-Werewolf")
print(f"[DEGEN] games where all scored players share one role: {len(degen)}")
for g, n, ro in degen:
    print(f"   {g}: {n} scored, all '{ro}'")
print(f"[ALLWOLF] games where every scored player == Werewolf: "
      f"{len(allwolf)}  {allwolf}")
