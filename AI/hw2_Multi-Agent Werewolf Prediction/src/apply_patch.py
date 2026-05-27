"""Apply manual private-case patches onto vC1_040 → new submission.

Reads annotations either from the audit sheet (filled-in ACTION: lines) or a
patches.csv (game,character,action). Applies deterministic transforms,
clamps, validates format (397 rows / 5 cols / no NaN), writes a submission.

Usage:
  python apply_patch.py --name manual_wolf_safe                 # parse sheet
  python apply_patch.py --name X --patches patches.csv          # parse csv
"""
from __future__ import annotations

import argparse, csv, re
from pathlib import Path
import pandas as pd

BASE = Path("artifacts/submissions/vC1_040_private.csv")
OUT_DIR = Path("artifacts/submissions")

ROLE_MAP = {"ROLE_MEDIUM": "Medium", "ROLE_SEER": "Seer",
            "ROLE_HUNTER": "Hunter", "ROLE_MADMAN": "Madman",
            "ROLE_WEREWOLF": "Werewolf", "ROLE_VILLAGER": "Villager"}
WS_OPS = {
    "BOOST_WOLF_SMALL": lambda w: min(1.0, w + 0.10),
    "BOOST_WOLF_BIG":   lambda w: min(1.0, w + 0.25),
    "DEMOTE_WOLF_SMALL": lambda w: max(0.0, w - 0.10),
    "DEMOTE_WOLF_BIG":   lambda w: max(0.0, w - 0.25),
    "SET_MIN_080": lambda w: max(w, 0.80),
    "SET_MAX_015": lambda w: min(w, 0.15),
    "SET_MAX_005": lambda w: min(w, 0.05),
}
VALID = set(WS_OPS) | set(ROLE_MAP) | {"KEEP"}

SHEET_GAME = re.compile(r"^##\s+(private_\d+)")
SHEET_SUSPECT = re.compile(r"^\s*\d+\.\s+\*\*(.+?)\*\*")
SHEET_ACTION = re.compile(r"^\s*ACTION:\s*(.+?)\s*$")


def parse_sheet(path: Path):
    game = cur = None
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = SHEET_GAME.match(ln)
        if m:
            game = m.group(1); cur = None; continue
        m = SHEET_SUSPECT.match(ln)
        if m:
            cur = m.group(1); continue
        m = SHEET_ACTION.match(ln)
        if m and game and cur:
            raw = m.group(1).replace("_", "_").strip()
            if raw and raw not in ("____", "_", "-"):
                for tok in re.split(r"[ ,+]+", raw):
                    tok = tok.strip().upper()
                    if tok in VALID and tok != "KEEP":
                        out.append((game, cur, tok))
            cur = None
    return out


def parse_csv(path: Path):
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            g = r["game"].strip()
            if not g.startswith("private_"):
                g = f"private_{g.zfill(2)}"
            for tok in re.split(r"[ ,+]+", r["action"].strip().upper()):
                if tok in VALID and tok != "KEEP":
                    out.append((g, r["character"].strip(), tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sheet", type=Path, default=Path("private_audit_sheet.md"))
    ap.add_argument("--patches", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patches = (parse_csv(args.patches) if args.patches
               else parse_sheet(args.sheet))
    if not patches:
        print("No patches parsed — nothing to do.")
        return

    df = pd.read_csv(BASE, dtype={"index": str})
    df["index"] = df["index"].str.zfill(2)
    df["_g"] = "private_" + df["index"]

    changed, miss, cnt = 0, [], {}
    for game, char, act in patches:
        mask = (df["_g"] == game) & (df["character"] == char)
        if not mask.any():
            miss.append((game, char, act)); continue
        i = df.index[mask][0]
        if act in WS_OPS:
            old = float(df.at[i, "wolf_score"])
            df.at[i, "wolf_score"] = WS_OPS[act](old)
        elif act in ROLE_MAP:
            df.at[i, "role"] = ROLE_MAP[act]
        changed += 1
        cnt[act] = cnt.get(act, 0) + 1
        print(f"  {game} {char}: {act}")

    if miss:
        print(f"\n[WARN] {len(miss)} unmatched (check exact name):")
        for m in miss:
            print(f"   {m}")

    out = df.drop(columns=["_g"]).copy()
    out["wolf_score"] = out["wolf_score"].clip(0.0, 1.0)
    assert len(out) == 397, f"row count {len(out)} != 397"
    assert list(out.columns) == ["id", "index", "character", "role", "wolf_score"]
    assert out["role"].notna().all() and out["wolf_score"].notna().all()
    assert out["role"].isin(["Villager", "Werewolf", "Seer", "Medium",
                             "Hunter", "Madman"]).all()

    print(f"\nApplied {changed} patches: {cnt}")
    if args.dry_run:
        print("(dry-run, not written)")
        return
    dst = OUT_DIR / f"{args.name}_private.csv"
    out.to_csv(dst, index=False)
    print(f"Wrote {dst}  ({changed} rows differ from vC1_040)")


if __name__ == "__main__":
    main()
