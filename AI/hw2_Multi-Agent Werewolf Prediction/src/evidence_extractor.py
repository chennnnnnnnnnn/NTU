"""Phase 2: extract structured evidence from parsed posts.

Inputs:
    - posts (list[Post] from parsers/posts.py)
    - csv_players (list[str] from roles.csv for this game)
    - speaker_map (dict[log_speaker -> csv_player] from parsers/names.py)

Outputs (dict):
    {
        "co_seer":           set of csv_player who claimed Seer
        "co_medium":         set of csv_player who claimed Medium
        "co_hunter":         set of csv_player who claimed Hunter
        "not_seer_medium":   set of csv_player who explicitly disclaim Seer & Medium
        "seer_blacks":       dict {csv_player -> set of TARGETS}  # who each claimed Seer called wolf
        "seer_whites":       dict {csv_player -> set of TARGETS}
        "medium_blacks":     same for medium
        "medium_whites":     same for medium
        "night_deaths":      list of csv_player (in chronological order)
        "executions":        list of csv_player
    }
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from parsers.posts import Post

# --- CO patterns ---
SEER_CO_RE = re.compile(
    r"\[\s*Seer\s*\]\s*:"                       # [Seer]:
    r"|\bCO\s+Seer\b"                            # CO Seer
    r"|I[\'’]?m\s+(?:the\s+|a\s+)?Seer\b"        # I'm the Seer
    r"|I\s+am\s+(?:the\s+|a\s+)?Seer\b",         # I am the Seer
    re.IGNORECASE,
)
MEDIUM_CO_RE = re.compile(
    r"\[\s*Medium\s*\]\s*:"
    r"|\bCO\s+Medium\b"
    r"|I[\'’]?m\s+(?:the\s+|a\s+)?Medium\b"
    r"|I\s+am\s+(?:the\s+|a\s+)?Medium\b",
    re.IGNORECASE,
)
HUNTER_CO_RE = re.compile(
    r"\[\s*Hunter\s*\]\s*:"
    r"|\bCO\s+Hunter\b"
    r"|I[\'’]?m\s+(?:the\s+|a\s+)?Hunter\b"
    r"|I\s+am\s+(?:the\s+|a\s+)?Hunter\b",
    re.IGNORECASE,
)
NOT_SEER_MED_RE = re.compile(
    r"not\s+(?:a\s+|the\s+)?Seer\s*(?:/|or)\s*not\s+(?:a\s+|the\s+)?Medium"
    r"|not\s+Seer\s*/\s*not\s+Medium"
    r"|\[\s*not\s+Seer\b.*?not\s+Medium\s*\]",
    re.IGNORECASE,
)

# --- divine result patterns (target placeholder will be substituted) ---
# We only run these on posts authored by a claimed Seer / Medium.
BLACK_WORDS = r"(?:wolf|werewolf|black)"
WHITE_WORDS = r"(?:human|white)"


# --- event patterns (carry-over from v1) ---
NIGHT_DEATH_RE = re.compile(r"The next morning,\s*(.+?)\s+was found in a gruesome state\.")
EXECUTION_RE = re.compile(r"^(.+?)\s+was executed by the villagers\.")


STOPWORDS = {"the"}


def _tokens(name: str) -> Set[str]:
    normalized = name.replace(",", " ").replace("_", " ")
    return {t.lower() for t in normalized.split() if t.lower() not in STOPWORDS}


def build_player_aliases(
    csv_players: List[str],
    speaker_map: Dict[str, str],
) -> Dict[str, Set[str]]:
    """For each csv_player, build the set of lowercase tokens/phrases that
    refer to them — used for divine-result target matching.

    Includes:
      - the csv name itself (lowercased)
      - every log speaker mapped to this player (lowercased)
      - tokens of the csv name UNIQUE to this player among the game roster
        (this gives us "gerd" alone, "jimzon", etc.)
    """
    # Reverse map: csv_player -> list of log speakers
    player_speakers: Dict[str, List[str]] = defaultdict(list)
    for sp, pl in speaker_map.items():
        player_speakers[pl].append(sp)

    # Count tokens by DISTINCT player ownership (csv + speakers form one bag per player).
    player_tokens: Dict[str, Set[str]] = {}
    for pl in csv_players:
        own = _tokens(pl) | {t for sp in player_speakers.get(pl, []) for t in _tokens(sp)}
        player_tokens[pl] = own

    token_player_count: Dict[str, int] = defaultdict(int)
    for pl, toks in player_tokens.items():
        for t in toks:
            token_player_count[t] += 1

    aliases: Dict[str, Set[str]] = {}
    for pl in csv_players:
        unique_tokens = {t for t in player_tokens[pl] if token_player_count[t] == 1}
        als = {pl.lower()}
        for sp in player_speakers.get(pl, []):
            als.add(sp.lower())
        als |= unique_tokens
        aliases[pl] = als
    return aliases


def _build_divine_pattern(alias: str, polarity: str) -> re.Pattern:
    """Compile a regex matching '<alias> is (a) <wolf-words>' or bracketed forms."""
    words = BLACK_WORDS if polarity == "black" else WHITE_WORDS
    a = re.escape(alias)
    # Two forms: bracketed "[X is wolf.]" and plain "X is a wolf"
    pat = (
        rf"\b{a}\b\s+(?:is|was)\s+(?:a\s+)?{words}\b"
        rf"|\[\s*{a}\b[^\]]{{0,40}}\b(?:is|was)\s+(?:a\s+)?{words}\b"
    )
    return re.compile(pat, re.IGNORECASE)


def extract_events_basic(log_text: str, players: List[str]) -> tuple:
    """Carry-over from v1: night_deaths and executions (substring match on csv_player)."""
    night_deaths: List[str] = []
    executions: List[str] = []
    player_set = set(players)

    for line in log_text.splitlines():
        line_str = line.strip()
        m = NIGHT_DEATH_RE.search(line_str)
        if m and m.group(1) in player_set:
            night_deaths.append(m.group(1))
            continue
        m = EXECUTION_RE.match(line_str)
        if m and m.group(1) in player_set:
            executions.append(m.group(1))
    return night_deaths, executions


def _find_divine_targets(
    text: str,
    aliases: Dict[str, Set[str]],
    polarity: str,
) -> Set[str]:
    targets = set()
    for player, als in aliases.items():
        for a in als:
            if len(a) < 3:
                continue  # avoid matching single letters
            pat = _build_divine_pattern(a, polarity)
            if pat.search(text):
                targets.add(player)
                break
    return targets


def extract_evidence(
    posts: List[Post],
    csv_players: List[str],
    speaker_map: Dict[str, str],
    log_text: str,
) -> dict:
    aliases = build_player_aliases(csv_players, speaker_map)

    # Per-player concatenated text (only posts BY this csv_player).
    posts_by_player: Dict[str, List[Post]] = defaultdict(list)
    for p in posts:
        csv = speaker_map.get(p.speaker)
        if csv:
            posts_by_player[csv].append(p)

    full_text_by_player: Dict[str, str] = {
        pl: "\n".join(p.text for p in ps)
        for pl, ps in posts_by_player.items()
    }

    co_seer: Set[str] = set()
    co_medium: Set[str] = set()
    co_hunter: Set[str] = set()
    not_seer_medium: Set[str] = set()

    for pl, full in full_text_by_player.items():
        if SEER_CO_RE.search(full):
            co_seer.add(pl)
        if MEDIUM_CO_RE.search(full):
            co_medium.add(pl)
        if HUNTER_CO_RE.search(full):
            co_hunter.add(pl)
        if NOT_SEER_MED_RE.search(full):
            not_seer_medium.add(pl)

    # Heuristic: if a player CO'd Seer/Medium/Hunter, ignore their "not Seer / not Medium"
    not_seer_medium -= (co_seer | co_medium)

    # Divine results: scan posts AUTHORED by claimed Seers / Mediums.
    seer_blacks: Dict[str, Set[str]] = {}
    seer_whites: Dict[str, Set[str]] = {}
    for seer in co_seer:
        text = full_text_by_player.get(seer, "")
        seer_blacks[seer] = _find_divine_targets(text, aliases, "black") - {seer}
        seer_whites[seer] = _find_divine_targets(text, aliases, "white") - {seer}

    medium_blacks: Dict[str, Set[str]] = {}
    medium_whites: Dict[str, Set[str]] = {}
    for med in co_medium:
        text = full_text_by_player.get(med, "")
        medium_blacks[med] = _find_divine_targets(text, aliases, "black") - {med}
        medium_whites[med] = _find_divine_targets(text, aliases, "white") - {med}

    night_deaths, executions = extract_events_basic(log_text, csv_players)

    return {
        "co_seer": co_seer,
        "co_medium": co_medium,
        "co_hunter": co_hunter,
        "not_seer_medium": not_seer_medium,
        "seer_blacks": seer_blacks,
        "seer_whites": seer_whites,
        "medium_blacks": medium_blacks,
        "medium_whites": medium_whites,
        "night_deaths": night_deaths,
        "executions": executions,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pandas as pd
    from parsers.posts import parse_posts
    from parsers.names import build_speaker_map

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=["public", "private"])
    ap.add_argument("--game", required=True, type=str, help="Game index, e.g. 01")
    args = ap.parse_args()

    df = pd.read_csv(args.data_dir / args.split / "roles.csv", dtype={"index": str})
    df["index"] = df["index"].str.zfill(2)
    players = df[df["index"] == args.game]["character"].tolist()
    text = (args.data_dir / args.split / f"{args.game}.txt").read_text(encoding="utf-8", errors="replace")
    posts = parse_posts(text, players)
    speaker_map = build_speaker_map(players, {p.speaker for p in posts})

    ev = extract_evidence(posts, players, speaker_map, text)
    for k, v in ev.items():
        print(f"{k:>18}: {v}")
