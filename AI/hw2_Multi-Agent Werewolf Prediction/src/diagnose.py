"""Error diagnosis on a public submission CSV.

Usage:
    python diagnose.py --pred artifacts/submissions/v7_public.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
)

ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Hunter", "Madman"]
GT_PATH = Path("/home/htiintern2502/AI2/data/Werewolf_Prediction_Dataset/public/roles_with_gt.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", type=Path, default=GT_PATH)
    args = ap.parse_args()

    pred = pd.read_csv(args.pred, dtype={"index": str})
    gt = pd.read_csv(args.gt, dtype={"index": str})
    m = pred.merge(
        gt[["id", "role", "wolf_score"]].rename(columns={"role": "role_gt", "wolf_score": "ws_gt"}),
        on="id",
    )

    print(f"=== Overall ({len(m)} rows, {(m['role_gt']=='Werewolf').sum()} true wolves) ===")
    macro = f1_score(m["role_gt"], m["role"], labels=ROLES, average="macro", zero_division=0)
    per_role = f1_score(m["role_gt"], m["role"], labels=ROLES, average=None, zero_division=0)
    wap = average_precision_score((m["role_gt"] == "Werewolf").astype(int), m["wolf_score"])
    final = 0.4 * macro + 0.6 * wap
    print(f"  Macro-F1: {macro:.4f}")
    print(f"  Wolf-AP:  {wap:.4f}")
    print(f"  Final:    {final:.4f}")
    print()
    print("Per-role F1:")
    for r, f in zip(ROLES, per_role):
        print(f"  {r:<10s} {f:.4f}")

    print()
    print("=== Confusion matrix (pred rows × GT cols) ===")
    cm = confusion_matrix(m["role_gt"], m["role"], labels=ROLES)
    # print with labels
    header = "          " + " ".join(f"{r[:6]:>6s}" for r in ROLES)
    print(header)
    for i, r in enumerate(ROLES):
        row = " ".join(f"{cm[i,j]:>6d}" for j in range(len(ROLES)))
        print(f"  GT {r[:6]:<7s} {row}")
    print("  (rows = GT, cols = predicted)")

    print()
    print("=== Per-GT-role error breakdown ===")
    for r in ROLES:
        gt_mask = m["role_gt"] == r
        n_total = gt_mask.sum()
        if n_total == 0:
            continue
        n_correct = ((m["role_gt"] == r) & (m["role"] == r)).sum()
        print(f"  {r:<10s} (n={n_total}, correct={n_correct})")
        wrong = m[gt_mask & (m["role"] != r)]
        if len(wrong) == 0:
            continue
        for pred_role, count in wrong["role"].value_counts().items():
            print(f"     → predicted as {pred_role:<10s} {count} times")

    print()
    print("=== Wolf-AP per game ===")
    per_game = []
    for game_idx, sub in m.groupby("index"):
        wolves = (sub["role_gt"] == "Werewolf").astype(int)
        if wolves.sum() > 0:
            ap = average_precision_score(wolves, sub["wolf_score"])
        else:
            ap = float("nan")
        # also compute Macro within this game
        per_game.append({
            "game": game_idx,
            "n_players": len(sub),
            "n_wolves": int(wolves.sum()),
            "wolf_ap": ap,
        })
    pg = pd.DataFrame(per_game).sort_values("wolf_ap")
    print(pg.to_string(index=False))

    print()
    print("=== Worst games (low Wolf-AP) — top 5 ===")
    worst = pg.head(5)
    for _, row in worst.iterrows():
        game_idx = row["game"]
        sub = m[m["index"] == game_idx]
        wolves = sub[sub["role_gt"] == "Werewolf"]
        pred_wolves = sub.nlargest(int(row["n_wolves"]), "wolf_score")
        print(f"\n  Game {game_idx} (AP={row['wolf_ap']:.3f}):")
        print(f"    True wolves: {list(wolves['character'])}")
        print(f"    Our top-{int(row['n_wolves'])}: {list(pred_wolves['character'])}")
        print(f"    Top wolf_score: {pred_wolves['wolf_score'].tolist()}")


if __name__ == "__main__":
    main()
