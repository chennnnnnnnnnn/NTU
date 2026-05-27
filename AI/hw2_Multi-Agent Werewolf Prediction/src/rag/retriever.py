"""Hybrid post retrieval per player (Phase 3.2).

For each player X, return a curated mix of posts that the LLM should read:

    1. Own posts                 — speaker == X        (cap 5)
    2. Posts mentioning X        — X ∈ mentioned       (cap 5)
    3. Reply chain               — posts in conversations with X
                                                       (cap 3)
    4. Embedding top-k           — query="suspicion/defense/accusation/vote
                                          /claim/contradiction about X"
                                                       (cap 5)

We dedupe by post_id and keep at most ~15 posts per player.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from parsers.posts import Post


@dataclass
class RetrievalCaps:
    own: int = 3
    mention: int = 3
    reply: int = 2
    embedding: int = 3
    max_chars_per_post: int = 200


def _short(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _format_post(p: Post, limit: int, csv_speaker: str | None) -> str:
    speaker = csv_speaker or p.speaker
    text = _short(p.text, limit)
    return f"[D{p.day} #{p.post_id} {speaker}] {text}"


def retrieve_for_player(
    target: str,
    posts: List[Post],
    speaker_map: Dict[str, str],
    aliases: Dict[str, set],
    caps: RetrievalCaps = RetrievalCaps(),
    embeddings: Optional[Dict[int, "np.ndarray"]] = None,
    embed_model=None,
) -> List[str]:
    """Return a list of formatted post strings for player `target`."""
    csv_to_speakers: Dict[str, List[str]] = defaultdict(list)
    for sp, pl in speaker_map.items():
        csv_to_speakers[pl].append(sp)
    target_speakers = set(csv_to_speakers.get(target, []))

    # Map post_id → post for replies lookup.
    by_id: Dict[int, Post] = {p.post_id: p for p in posts}

    used: set = set()
    selected: List[Post] = []

    def add(p: Post):
        if p.post_id in used:
            return
        used.add(p.post_id)
        selected.append(p)

    # 1. Own posts (most recent slice — wolves often slip up late)
    own = [p for p in posts if p.speaker in target_speakers]
    # Keep first 2 (intro-day reveals) + last (caps.own - 2) (more recent decisions).
    if len(own) > caps.own:
        keep = own[:2] + own[-(caps.own - 2):]
    else:
        keep = own
    for p in keep:
        add(p)

    # 2. Posts mentioning X (by csv name token or alias)
    target_aliases = aliases.get(target, set())
    mentioning = []
    for p in posts:
        if p.speaker in target_speakers:
            continue
        # Check log-name match (already filled by parsers/posts.parse_posts) or alias.
        if target in p.mentioned:
            mentioning.append(p)
            continue
        # Fall back to alias-substring on text.
        low = p.text.lower()
        if any(a for a in target_aliases if len(a) >= 4 and a in low):
            mentioning.append(p)
    # Prefer day-2+ posts (more informational).
    mentioning.sort(key=lambda p: (-p.day, p.post_id))
    for p in mentioning[: caps.mention]:
        add(p)

    # 3. Reply chain: posts that reply to one of X's posts, or X's replies to others.
    own_ids = {p.post_id for p in own}
    reply_chain = []
    for p in posts:
        if p.speaker in target_speakers:
            # X's reply pulls in the replied-to post too.
            for r in p.replied_to:
                if r in by_id:
                    reply_chain.append(by_id[r])
        elif any(r in own_ids for r in p.replied_to):
            reply_chain.append(p)
    # Dedupe and cap.
    seen_reply = set()
    reply_chain_clean = []
    for p in reply_chain:
        if p.post_id in seen_reply:
            continue
        seen_reply.add(p.post_id)
        reply_chain_clean.append(p)
    for p in reply_chain_clean[: caps.reply]:
        add(p)

    # 4. Embedding top-k (optional)
    if embeddings is not None and embed_model is not None and caps.embedding > 0:
        import numpy as np
        query = (
            f"suspicion accusation defense vote claim contradiction about {target}"
        )
        q = embed_model.encode([query], normalize_embeddings=True)[0]
        scored = []
        for pid, emb in embeddings.items():
            if pid in used:
                continue
            sim = float(np.dot(q, emb))
            scored.append((sim, pid))
        scored.sort(reverse=True)
        for _, pid in scored[: caps.embedding]:
            if pid in by_id:
                add(by_id[pid])

    # Sort final selection chronologically for readability.
    selected.sort(key=lambda p: p.post_id)

    out = []
    for p in selected:
        csv = speaker_map.get(p.speaker)
        out.append(_format_post(p, caps.max_chars_per_post, csv))
    return out
