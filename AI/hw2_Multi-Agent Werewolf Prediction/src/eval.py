"""Local evaluation: Macro-F1 + Wolf-AP + Final Score, per-role F1.

Usage:
    python eval.py \
        --pred artifacts/submissions/v1_public.csv \
        --gt   /path/to/roles_with_gt.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Hunter", "Madman"]


def evaluate(pred_path: Path, gt_path: Path) -> dict:
    pred = pd.read_csv(pred_path)
    gt = pd.read_csv(gt_path)

    merged = pred.merge(
        gt[["id", "role", "wolf_score"]].rename(
            columns={"role": "role_gt", "wolf_score": "wolf_score_gt"}
        ),
        on="id",
        how="inner",
    )
    if len(merged) != len(gt):
        print(
            f"[warn] joined {len(merged)} rows but GT has {len(gt)} "
            f"(pred has {len(pred)})"
        )

    y_true = merged["role_gt"].tolist()
    y_pred = merged["role"].tolist()

    macro_f1 = f1_score(y_true, y_pred, labels=ROLES, average="macro", zero_division=0)
    per_role_f1 = f1_score(y_true, y_pred, labels=ROLES, average=None, zero_division=0)

    wolf_gt = (merged["role_gt"] == "Werewolf").astype(int).to_numpy()
    wolf_pred = merged["wolf_score"].to_numpy()
    if wolf_gt.sum() == 0:
        wolf_ap = float("nan")
    else:
        wolf_ap = average_precision_score(wolf_gt, wolf_pred)

    final = 0.4 * macro_f1 + 0.6 * wolf_ap

    return {
        "macro_f1": macro_f1,
        "wolf_ap": wolf_ap,
        "final": final,
        "per_role_f1": dict(zip(ROLES, per_role_f1)),
        "n_rows": len(merged),
        "n_wolves_gt": int(wolf_gt.sum()),
    }


def print_report(r: dict) -> None:
    print(f"Macro-F1:    {r['macro_f1']:.4f}")
    print(f"Wolf-AP:     {r['wolf_ap']:.4f}")
    print(f"Final Score: {r['final']:.4f}")
    print()
    print("Per-role F1:")
    for role, f1 in r["per_role_f1"].items():
        print(f"  {role:<10s} {f1:.4f}")
    print()
    print(f"Joined rows: {r['n_rows']}, true wolves: {r['n_wolves_gt']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    args = ap.parse_args()
    print_report(evaluate(args.pred, args.gt))


if __name__ == "__main__":
    main()
