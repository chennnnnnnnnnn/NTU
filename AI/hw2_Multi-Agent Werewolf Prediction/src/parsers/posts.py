"""Parse a Werewolf game log into a list of post units.

Each post in the log looks like:

    N.
    Speaker Name
    HH:MM
    content line 1
    content line 2
    ...
    M.
    Next Speaker
    ...

Day markers like "===== Day 2 =====" partition the log into day phases.

Output schema (one dict per post):

    {
        "post_id":          int,   # the "N." number
        "day":              int,   # 0 = prologue / pre-Day-1
        "speaker":          str,
        "time":             str,   # "HH:MM"
        "text":             str,   # joined content lines
        "replied_to":       list[int],  # >>N references
        "mentioned":        list[str],  # known player names appearing in text
        "line_start":       int,   # 1-indexed line number of the "N." marker
        "line_end":         int,
    }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import List, Optional

POST_NUM_RE = re.compile(r"^\s*(\d+)\.\s*$")
TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")
DAY_RE = re.compile(r"^=+\s*Day\s+(\d+)\s*=+\s*$", re.IGNORECASE)
REPLY_RE = re.compile(r">>(\d+)")


@dataclass
class Post:
    post_id: int
    day: int
    speaker: str
    time: str
    text: str
    replied_to: List[int] = field(default_factory=list)
    mentioned: List[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


def _is_post_header(lines: List[str], i: int) -> Optional[tuple]:
    """If lines[i] starts a valid post header, return (post_id, speaker, time, content_start_index)."""
    m = POST_NUM_RE.match(lines[i])
    if not m:
        return None
    if i + 2 >= len(lines):
        return None
    speaker_line = lines[i + 1].strip()
    time_line = lines[i + 2].strip()
    if not speaker_line:
        return None
    if not TIME_RE.match(time_line):
        return None
    return (int(m.group(1)), speaker_line, time_line, i + 3)


def parse_posts(text: str, known_players: Optional[List[str]] = None) -> List[Post]:
    """Parse the log into a list of `Post`.

    `known_players` is used to (a) validate speaker names — speakers not in the
    list still produce posts but the field stays as-is, and (b) populate the
    `mentioned` field.
    """
    lines = text.splitlines()
    n = len(lines)
    player_set = set(known_players or [])

    # First pass: find all candidate post-header indices.
    headers = []
    i = 0
    while i < n:
        h = _is_post_header(lines, i)
        if h is not None:
            headers.append((i, *h))  # (line_index, post_id, speaker, time, content_start)
            i = h[3]
        else:
            i += 1

    posts: List[Post] = []
    current_day = 0
    for idx, (start_line, post_id, speaker, time, content_start) in enumerate(headers):
        end_line = headers[idx + 1][0] if idx + 1 < len(headers) else n
        content_lines = lines[content_start:end_line]
        text_block = "\n".join(content_lines).strip()

        # Day tracking: scan everything BEFORE this post's content_start for "Day N".
        # Process day markers within content too (some content has day end tags).
        for ln in lines[max(0, start_line - 0):end_line]:
            dm = DAY_RE.match(ln.strip())
            if dm:
                current_day = max(current_day, int(dm.group(1)))

        replied = [int(m) for m in REPLY_RE.findall(text_block)]
        mentioned = [p for p in player_set if p in text_block]

        posts.append(Post(
            post_id=post_id,
            day=current_day,
            speaker=speaker,
            time=time,
            text=text_block,
            replied_to=replied,
            mentioned=mentioned,
            line_start=start_line + 1,  # 1-indexed
            line_end=end_line,
        ))

    return posts


def to_dicts(posts: List[Post]) -> list:
    return [asdict(p) for p in posts]


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--players", type=str, default=None,
                    help="comma-separated known player names")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    players = args.players.split(",") if args.players else None
    posts = parse_posts(args.log.read_text(encoding="utf-8", errors="replace"), players)
    print(f"Parsed {len(posts)} posts")
    for p in posts[:args.limit]:
        print(json.dumps(asdict(p), ensure_ascii=False)[:200])
