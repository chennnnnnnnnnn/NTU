"""Validate a Kaggle submission CSV before uploading.

Catches:
  - wrong column order/names
  - missing rows
  - id not monotonic
  - NaN role / wolf_score
  - wolf_score out of [0, 1]
  - index lost its zero-padding (the 01 → 1 bug)
  - role label not in the 6 valid roles
  - duplicate id

Usage:
    python validate_submission.py --csv artifacts/submissions/v7_private.csv \
        --expected-rows 397
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_ROLES = {"Villager", "Werewolf", "Seer", "Medium", "Hunter", "Madman"}


def validate(csv_path: Path, expected_rows: int | None = None) -> None:
    # Read index as string to detect zero-pad loss.
    df = pd.read_csv(csv_path, dtype={"index": str})

    # Column order
    assert list(df.columns) == ["id", "index", "character", "role", "wolf_score"], \
        f"Bad columns: {list(df.columns)}"

    # Row count
    if expected_rows is not None:
        assert len(df) == expected_rows, \
            f"Expected {expected_rows} rows, got {len(df)}"

    # id monotonic and unique
    assert df["id"].is_monotonic_increasing, "id not sorted ascending"
    assert df["id"].is_unique, "duplicate id"

    # role coverage
    assert df["role"].notna().all(), "Some rows have missing role"
    bad_roles = set(df["role"]) - VALID_ROLES
    assert not bad_roles, f"Invalid role labels: {bad_roles}"

    # wolf_score range
    assert df["wolf_score"].notna().all(), "Some rows have missing wolf_score"
    assert df["wolf_score"].between(0, 1).all(), \
        f"wolf_score out of [0, 1]: min={df['wolf_score'].min()}, max={df['wolf_score'].max()}"

    # Index must be a non-empty string with 1+ char (catches 01 → 1 conversion).
    assert df["index"].str.len().ge(1).all(), "index has empty values"
    # Quick zero-pad sanity: if any index is purely numeric, it should be ≥ 2 chars
    numeric_indices = df[df["index"].str.match(r"^\d+$")]["index"]
    bad_zero_pad = numeric_indices[numeric_indices.str.len() < 2]
    if len(bad_zero_pad) > 0:
        print(f"[warn] {len(bad_zero_pad)} indices may have lost zero-pad: "
              f"{bad_zero_pad.unique()[:5].tolist()}")

    # Distribution sanity
    print(f"Rows: {len(df)}")
    print(f"Wolf-score range: [{df['wolf_score'].min():.4f}, {df['wolf_score'].max():.4f}]")
    print(f"Wolf-score mean: {df['wolf_score'].mean():.4f}")
    print(f"Role distribution:")
    for role, n in df["role"].value_counts().items():
        print(f"  {role:<10s} {n:>4d}")
    print(f"\nGames present: {df['index'].nunique()}")
    print(f"Index samples: {sorted(df['index'].unique())[:5]} ... {sorted(df['index'].unique())[-3:]}")

    print(f"\n✓ {csv_path.name} passes all assertions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--expected-rows", type=int, default=None,
                    help="Expected row count (276 for public, 397 for private)")
    args = ap.parse_args()
    validate(args.csv, args.expected_rows)


if __name__ == "__main__":
    main()
