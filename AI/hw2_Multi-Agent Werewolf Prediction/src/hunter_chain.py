"""Hunter-claim protection chain (rule-based, ROLE-ONLY).

Hunters rarely fake-claim and tend to use protective language. We scan each
player's posts for Hunter-signalling phrases and boost role_scores[Hunter]
only. wolf_score is never touched (AP line is saturated; role line has room).

Signals (regex, on the player's own posts):
  - explicit "[Hunter]" / "I am the Hunter" / "CO Hunter"
  - "don't lynch me" / "don't vote me" / "I have a role" / "I can prove"
  - "I'll take someone with me" / protect-style statements
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

from parsers.posts import parse_posts
from parsers.names import build_speaker_map

_HUNTER_STRONG = re.compile(
    r"\[\s*hunter\s*\]|\bco\s+hunter\b"
    r"|i[' ]?a?m\s+(?:the\s+|a\s+)?hunter\b"
    r"|i\s+am\s+(?:the\s+|a\s+)?hunter\b",
    re.IGNORECASE,
)
_HUNTER_SOFT = re.compile(
    r"don'?t\s+(?:lynch|vote|execute)\s+me"
    r"|i\s+have\s+(?:a\s+)?(?:role|ability|power)"
    r"|i\s+can\s+prove"
    r"|i'?ll\s+take\s+(?:someone|him|her|them|a\s+wolf)\s+with\s+me"
    r"|if\s+you\s+lynch\s+me",
    re.IGNORECASE,
)
_HUNTER_DISCLAIM = re.compile(
    r"(not|isn'?t|ain'?t|no)\s+(?:a\s+)?hunter\b|i\s+am\s+not\s+(?:the\s+)?hunter",
    re.IGNORECASE,
)


def apply_hunter_chain(
    players: list[str],
    wolf_score: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    log_text: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """ROLE-ONLY. Boost Hunter role_score from claim language."""
    posts = parse_posts(log_text, players)
    spk = build_speaker_map(players, {p.speaker for p in posts})

    by_player: Dict[str, list] = {p: [] for p in players}
    for po in posts:
        csv = spk.get(po.speaker)
        if csv in by_player:
            by_player[csv].append(po.text)

    rs = {p: dict(d) for p, d in role_scores.items()}
    for p, texts in by_player.items():
        if p not in rs:
            continue
        full = "\n".join(texts)
        if _HUNTER_DISCLAIM.search(full) and not _HUNTER_STRONG.search(full):
            continue
        strong = bool(_HUNTER_STRONG.search(full))
        soft = len(_HUNTER_SOFT.findall(full))
        boost = 0.0
        if strong:
            boost += 0.45
        if soft:
            boost += min(soft, 3) * 0.10
        if boost > 0:
            rs[p]["Hunter"] = rs[p].get("Hunter", 0.0) + boost
            # A genuine Hunter claimer is less likely Werewolf/Madman
            rs[p]["Werewolf"] = rs[p].get("Werewolf", 0.0) - 0.10 * boost
    return wolf_score, rs
