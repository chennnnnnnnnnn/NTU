"""Resolve speaker names in logs to the canonical names used in roles.csv.

Three observed formats:
    - "Optimist Gerd"           (most public games)
    - "Gerd"                    (e.g. public/07, private/30)
    - "Gerd the Optimist"       (e.g. private/24)

The proper name (Gerd, Peter, ...) is the unique identifier shared across
formats. We resolve by token-overlap: each candidate is matched to the
csv-player name sharing the most tokens.
"""
from __future__ import annotations

from typing import Dict, Iterable

STOPWORDS = {"the"}


def _tokens(name: str) -> set:
    # Lower-case matching: some games use UPPERCASE names (e.g. public/09 "KATHARINA")
    # vs the log's title-case ("Shepherd Katharina"). Also split on underscore so
    # "FATHER_JIMZON" → {"father", "jimzon"}.
    normalized = name.replace(",", " ").replace("_", " ")
    return {t.lower() for t in normalized.split() if t.lower() not in STOPWORDS}


def build_speaker_map(
    csv_players: Iterable[str],
    log_speakers: Iterable[str],
) -> Dict[str, str]:
    """Return {log_speaker: csv_player}.

    Match score is Jaccard-style:
        overlap_ratio = |sp ∩ pl| / min(|sp|, |pl|)
    This prefers exact-name matches over loose overlaps. E.g. "Village Girl
    Pamela" matches CSV "PAMELA" (1/1=1.0) rather than "Young Girl Liza"
    (1/3=0.33).

    Ties broken deterministically by csv-player name.
    """
    csv_players = list(csv_players)
    out: Dict[str, str] = {}
    csv_token_sets = {pl: _tokens(pl) for pl in csv_players}
    for sp in log_speakers:
        sp_tokens = _tokens(sp)
        if not sp_tokens:
            continue
        best_player = None
        best_score = 0.0
        for pl in sorted(csv_token_sets):
            pl_tokens = csv_token_sets[pl]
            if not pl_tokens:
                continue
            overlap = len(sp_tokens & pl_tokens)
            if overlap == 0:
                continue
            score = overlap / min(len(sp_tokens), len(pl_tokens))
            if score > best_score:
                best_score = score
                best_player = pl
        if best_player is not None:
            out[sp] = best_player
    return out
