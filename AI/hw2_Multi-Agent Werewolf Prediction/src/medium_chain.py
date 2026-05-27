"""Medium-execution chain scorer (rule-based, low overfit risk).

Game-rule logic:
  - executions are public ground truth (regex-extracted, ordered)
  - a real Medium only learns alignment of EXECUTED (or dead) players
  - a claimed Medium whose reports target executed players (not living ones)
    is more likely the real Medium → medium_confidence
  - a trusted Medium's "werewolf" calls → soft wolf_score raise
                       "human" calls    → soft wolf_score lower

Soft confirmation (per user spec) — never hard-set 1.0:
  werewolf: wolf_score = max(ws, 0.60 + 0.30 * conf)
  human:    wolf_score = min(ws, 0.40 - 0.25 * conf)

Also feeds role_scores (Medium up for trusted reporter; Werewolf up/down for
targets) but role solver stays independent of wolf_score.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def medium_confidence(
    reporter: str,
    med_reports: Dict[str, list],
    executed: set,
    night_deaths: set,
    co_medium: set,
    not_seer_medium: set,
) -> float:
    """Score how likely `reporter` is the real Medium, → [0, 1]."""
    score = 0.0
    targets = set(med_reports.get("wolf", [])) | set(med_reports.get("human", []))
    if not targets:
        return 0.0
    for t in targets:
        if t in executed:
            score += 2.0          # medium correctly talks about executed players
        elif t in night_deaths:
            score += 0.5          # medium usually sees executed, not night-dead
        else:
            score -= 2.0          # reported a LIVING player → not a real medium
    if reporter in co_medium:
        score += 1.0
    if reporter in not_seer_medium:
        score -= 1.5              # disclaimed Medium but "reports" → inconsistent
    # Scale so a clean single-execution report (+2 +1 = 3) → conf ≈ 0.82
    return _sigmoid(score / 2.0)


CHAIN_SCALE = 1.0  # global soft-weight for chain role deltas

# vC1: soft game-rule wolf injection. 0.0 = off (vB1 behaviour).
WOLF_SOFT_NUDGE = 0.0     # vC1 additive: ws += WOLF_SOFT_NUDGE * conf
WOLF_SOFT_CONF_MIN = 0.8  # only fire when medium_confidence is very high
WOLF_SOFT_CAP = 0.0       # if >0: ws = max(ws, CAP) instead of additive nudge
WOLF_HUMAN_DECAY = 0.0    # vC4: ws -= WOLF_HUMAN_DECAY * conf for medium-confirmed human
VC7_WHITELIST = {}        # vC7: {(split,game_idx): [(target, nudge), ...]} forced fires
VC7_GAME = None           # set per game by main.py: "split/idx"


def apply_medium_chain(
    players: list[str],
    wolf_score: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    results: Dict,
    executions: list[str],
    night_deaths: list[str],
    co_medium: set,
    not_seer_medium: set,
    affect_wolf: bool = False,   # hard wolf path (legacy, kept off)
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return updated (wolf_score, role_scores)."""
    executed = set(executions)
    nd = set(night_deaths)
    med_by = results.get("medium_by", {}) or {}

    ws = dict(wolf_score)
    rs = {p: dict(d) for p, d in role_scores.items()}

    for reporter, rep in med_by.items():
        if reporter not in ws:
            continue
        conf = medium_confidence(reporter, rep, executed, nd,
                                 set(co_medium), set(not_seer_medium))
        if conf < 0.5:
            continue  # not trustworthy enough to act on

        # Role: trusted reporter looks like Medium
        if reporter in rs:
            rs[reporter]["Medium"] = rs[reporter].get("Medium", 0.0) + CHAIN_SCALE * 0.5 * conf
            rs[reporter]["Madman"] = rs[reporter].get("Madman", 0.0) - CHAIN_SCALE * 0.2 * conf
            rs[reporter]["Werewolf"] = rs[reporter].get("Werewolf", 0.0) - CHAIN_SCALE * 0.2 * conf

        for tgt in rep.get("wolf", []):
            if tgt not in ws:
                continue
            if tgt not in executed and tgt not in nd:
                continue  # only trust reports about dead players
            if affect_wolf:
                ws[tgt] = max(ws[tgt], 0.60 + 0.30 * conf)
            # vC1: SOFT game-rule wolf injection — high-conf medium, executed
            # target, wolf verdict → small additive nudge (NOT hard overwrite).
            if conf >= WOLF_SOFT_CONF_MIN and tgt in executed:
                if WOLF_SOFT_CAP > 0.0:
                    ws[tgt] = max(ws[tgt], WOLF_SOFT_CAP)
                elif WOLF_SOFT_NUDGE > 0.0:
                    ws[tgt] = min(1.0, ws[tgt] + WOLF_SOFT_NUDGE * conf)
            if tgt in rs:
                rs[tgt]["Werewolf"] = rs[tgt].get("Werewolf", 0.0) + CHAIN_SCALE * 0.4 * conf

        for tgt in rep.get("human", []):
            if tgt not in ws:
                continue
            if tgt not in executed and tgt not in nd:
                continue
            if affect_wolf:
                ws[tgt] = min(ws[tgt], 0.40 - 0.25 * conf)
            # vC4: high-conf Medium says executed target is HUMAN → not a wolf
            # → soft-DECAY wolf_score (same deterministic anchor as vC1).
            if (WOLF_HUMAN_DECAY > 0.0
                    and conf >= WOLF_SOFT_CONF_MIN
                    and tgt in executed):
                ws[tgt] = max(0.0, ws[tgt] - WOLF_HUMAN_DECAY * conf)
            if tgt in rs:
                rs[tgt]["Werewolf"] = rs[tgt].get("Werewolf", 0.0) - CHAIN_SCALE * 0.4 * conf
                rs[tgt]["Villager"] = rs[tgt].get("Villager", 0.0) + CHAIN_SCALE * 0.2 * conf

    if VC7_GAME in VC7_WHITELIST:
        for tgt, nud in VC7_WHITELIST[VC7_GAME]:
            if tgt in ws:
                ws[tgt] = min(1.0, ws[tgt] + nud)
    ws = {p: max(0.0, min(1.0, v)) for p, v in ws.items()}
    return ws, rs


def load_results(results_dir: Path, split: str, game_idx: str) -> Dict:
    p = results_dir / f"{split}_{game_idx}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())
