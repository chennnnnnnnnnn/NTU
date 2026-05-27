"""Driver: load model once, iterate over evidence JSONs, dump predictions.

Phase 3.1 (evidence-only): no --use-rag, no embeddings.
Phase 3.2 (RAG):           --use-rag --embeddings-dir artifacts/embeddings

Usage:
    python run_llm.py \
        --model models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
        --evidence-dir artifacts/evidence \
        --output-dir artifacts/llm_pred_rag \
        --raw-dir artifacts/llm_raw_rag \
        --games public_01 public_02 public_03 \
        --ngl -1 \
        --use-rag \
        --data-dir /home/htiintern2502/AI2/data/Werewolf_Prediction_Dataset \
        --embeddings-dir artifacts/embeddings
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List

from agents.reasoner import Reasoner, ReasonerConfig, run_one_game


def _retrieve_for_game(record: Dict, data_dir: Path, embeddings_dir: Path | None, embed_model) -> Dict[str, List[str]]:
    """Build retrieved-posts dict {player: [formatted post strings]}."""
    from parsers.posts import parse_posts
    from parsers.names import build_speaker_map
    from evidence_extractor import build_player_aliases
    from rag.retriever import retrieve_for_player, RetrievalCaps

    split = record["split"]
    game_idx = record["game_idx"]
    players = record["players_csv"]
    text = (data_dir / split / f"{game_idx}.txt").read_text(encoding="utf-8", errors="replace")
    posts = parse_posts(text, players)
    speaker_map = build_speaker_map(players, {p.speaker for p in posts})
    aliases = build_player_aliases(players, speaker_map)

    embeddings = None
    if embeddings_dir is not None and embed_model is not None:
        emb_path = embeddings_dir / f"{split}_{game_idx}.pkl"
        if emb_path.exists():
            with open(emb_path, "rb") as f:
                embeddings = pickle.load(f)

    caps = RetrievalCaps()
    out = {}
    for p in players:
        out[p] = retrieve_for_player(
            target=p, posts=posts, speaker_map=speaker_map, aliases=aliases,
            caps=caps, embeddings=embeddings, embed_model=embed_model,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--evidence-dir", type=Path, default=Path("artifacts/evidence"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/llm_pred"))
    ap.add_argument("--raw-dir", type=Path, default=Path("artifacts/llm_raw"))
    ap.add_argument("--games", nargs="+", required=True,
                    help='Game ids like "public_01 public_02"')
    ap.add_argument("--ngl", type=int, default=-1, help="GPU layers (-1 = all)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    # RAG flags
    ap.add_argument("--use-rag", action="store_true",
                    help="Use RAG-2 retrieval (Phase 3.2)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Werewolf_Prediction_Dataset dir (required if --use-rag)")
    ap.add_argument("--embeddings-dir", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--embed-model", type=str,
                    default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--variant", default="general",
                    choices=["general", "voting", "madman"],
                    help="Prompt variant for self-consistency")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReasonerConfig(
        model_path=args.model,
        n_gpu_layers=args.ngl,
        n_ctx=args.ctx,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(f"Loading LLM: {args.model} (ngl={args.ngl}, ctx={args.ctx})")
    t0 = time.time()
    reasoner = Reasoner(cfg)
    print(f"LLM loaded in {time.time()-t0:.1f}s")

    embed_model = None
    if args.use_rag:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required with --use-rag")
        print(f"Loading embed model {args.embed_model}…")
        t0 = time.time()
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(args.embed_model, device="cuda")
        print(f"Embed model loaded in {time.time()-t0:.1f}s")

    n_ok = 0
    n_fail = 0
    for game in args.games:
        ev_path = args.evidence_dir / f"{game}.json"
        if not ev_path.exists():
            print(f"[warn] {ev_path} missing, skip")
            continue
        record = json.loads(ev_path.read_text())

        retrieved = None
        if args.use_rag:
            retrieved = _retrieve_for_game(record, args.data_dir, args.embeddings_dir, embed_model)

        t1 = time.time()
        preds = run_one_game(
            reasoner, record,
            raw_save_path=args.raw_dir / f"{game}.txt",
            retrieved_posts=retrieved,
            variant=args.variant,
        )
        dt = time.time() - t1
        if preds is None:
            print(f"  [FAIL] {game}  ({dt:.1f}s) — JSON parse failed")
            n_fail += 1
            continue
        out_path = args.output_dir / f"{game}.predictions.json"
        out_path.write_text(json.dumps(preds, indent=2, ensure_ascii=False))
        n_ok += 1
        top3 = sorted([(p, round(d['wolf_score'], 2)) for p, d in preds.items()], key=lambda x: -x[1])[:3]
        n_madman = sum(1 for d in preds.values() if d.get("madman_like"))
        print(f"  [OK]   {game}  ({dt:.1f}s) — {len(preds)} players, madman_like={n_madman}, top wolves: {top3}")

    print(f"\nTotal: {n_ok} ok, {n_fail} failed.")


if __name__ == "__main__":
    main()
