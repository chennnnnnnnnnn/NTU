"""Pre-compute sentence-transformer embeddings for every post in every game.

Output:  artifacts/embeddings/{split}_{game}.pkl
         dict {post_id: np.ndarray (384-dim)}

Usage:
    python -m rag.build_index \
        --data-dir /home/htiintern2502/AI2/data/Werewolf_Prediction_Dataset \
        --output-dir artifacts/embeddings
"""
from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from parsers.names import build_speaker_map
from parsers.posts import parse_posts


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _clean(text: str, limit: int = 400) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) > limit:
        text = text[:limit]
    return text


def build_for_game(
    text: str,
    players: list[str],
    embed_model,
    batch_size: int = 64,
) -> dict:
    posts = parse_posts(text, players)
    # Optionally enrich with speaker name in the embedded text so retrieval
    # for a player works better.
    speaker_map = build_speaker_map(players, {p.speaker for p in posts})
    inputs = []
    pids = []
    for p in posts:
        csv = speaker_map.get(p.speaker, p.speaker)
        # Prepend speaker to give embedding semantic context.
        s = f"{csv} (Day {p.day}): {_clean(p.text)}"
        inputs.append(s)
        pids.append(p.post_id)
    if not inputs:
        return {}
    vecs = embed_model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {pid: vec.astype(np.float32) for pid, vec in zip(pids, vecs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--splits", nargs="+", default=["public", "private"])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    total_t = 0
    for split in args.splits:
        roles_csv = args.data_dir / split / "roles.csv"
        df = pd.read_csv(roles_csv, dtype={"index": str})
        df["index"] = df["index"].str.zfill(2)
        for idx, sub in df.groupby("index", sort=True):
            players = sub["character"].tolist()
            log_path = args.data_dir / split / f"{idx}.txt"
            text = log_path.read_text(encoding="utf-8", errors="replace")
            t0 = time.time()
            embs = build_for_game(text, players, model)
            dt = time.time() - t0
            total_t += dt
            out_path = args.output_dir / f"{split}_{idx}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(embs, f)
            print(f"  {split}/{idx}: {len(embs)} posts indexed in {dt:.1f}s")
    print(f"\nTotal embedding time: {total_t:.1f}s")


if __name__ == "__main__":
    main()
