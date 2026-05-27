"""Seer-divination chain (rule-based, ROLE-ONLY, conservative).

Key game-rule differences vs Medium:
  - Seer divines LIVING players (not executed) → no "target in executed" filter.
  - Seers are faked far more than Mediums → be conservative; cross-check with
    the trusted Medium's reveals.

seer_confidence(reporter) raised by:
  + has a Seer CO
  + multiple divination results (acting consistently as Seer)
  + results NOT self-contradictory (never says X human then X wolf)
lowered by:
  - self-contradiction
  - contradicted by trusted Medium's reveals (Seer said wolf, Medium said human)
  - disclaimed Seer/Medium but still "reports"
  - many claimed Seers (each individual claimer less likely real)

ROLE-ONLY: adjusts role_scores[Seer]/[Madman] only. Never touches wolf_score
(go/no-go: AP line is saturated; role line still has headroom).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


CHAIN_SCALE = 1.0
SEER_WOLF_NUDGE = 0.0       # vC2 additive wolf nudge (0 = off)
SEER_WOLF_CONF_MIN = 0.90  # stricter than medium (seer faked more)


def seer_confidence(
    reporter: str,
    div_reports: Dict[str, list],
    co_seer: set,
    not_seer_medium: set,
    n_claimed_seers: int,
    trusted_medium_wolf: set,
    trusted_medium_human: set,
) -> float:
    wolves = set(div_reports.get("wolf", []))
    humans = set(div_reports.get("human", []))
    if not (wolves or humans):
        return 0.0
    score = 0.0

    n_results = len(wolves) + len(humans)
    score += min(n_results, 4) * 0.6          # acting as Seer over several calls

    # self-contradiction (same target both wolf and human)
    contradiction = wolves & humans
    score -= 2.0 * len(contradiction)

    # cross-check with trusted Medium
    score -= 2.0 * len(wolves & trusted_medium_human)   # Seer said wolf, Medium said human
    score -= 2.0 * len(humans & trusted_medium_wolf)    # Seer said human, Medium said wolf
    score += 1.5 * len(wolves & trusted_medium_wolf)    # agreement → real Seer
    score += 1.5 * len(humans & trusted_medium_human)

    if reporter in co_seer:
        score += 1.0
    if reporter in not_seer_medium:
        score -= 1.5
    # More claimed Seers → each individual less likely the real one
    if n_claimed_seers >= 2:
        score -= 0.8 * (n_claimed_seers - 1)

    return _sigmoid(score / 2.0)


def apply_seer_chain(
    players: list[str],
    wolf_score: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    results: Dict,
    co_seer: set,
    not_seer_medium: set,
    trusted_medium_wolf: set,
    trusted_medium_human: set,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """ROLE-ONLY. Returns (wolf_score unchanged, role_scores updated)."""
    div_by = results.get("divination_by", {}) or {}
    rs = {p: dict(d) for p, d in role_scores.items()}
    n_claimed = len(div_by)

    confs = {}
    for reporter, rep in div_by.items():
        if reporter not in rs:
            continue
        confs[reporter] = seer_confidence(
            reporter, rep, set(co_seer), set(not_seer_medium),
            n_claimed, trusted_medium_wolf, trusted_medium_human,
        )

    if not confs:
        return wolf_score, rs

    # The single most-confident reporter → real Seer; others (if they claimed)
    # → likely Madman / fake.
    best = max(confs, key=confs.get)
    best_conf = confs[best]

    ws = dict(wolf_score)

    if best_conf >= 0.55:
        rs[best]["Seer"] = rs[best].get("Seer", 0.0) + CHAIN_SCALE * 0.55 * best_conf
        rs[best]["Madman"] = rs[best].get("Madman", 0.0) - CHAIN_SCALE * 0.15 * best_conf
        rs[best]["Werewolf"] = rs[best].get("Werewolf", 0.0) - CHAIN_SCALE * 0.15 * best_conf

    # vC2: Seer-divination wolf injection. STRICTER than medium (seer is faked
    # far more): only the single most-trusted Seer, conf >= SEER_WOLF_CONF_MIN,
    # and target NOT contradicted by trusted Medium's human verdict.
    if (SEER_WOLF_NUDGE > 0.0
            and best_conf >= SEER_WOLF_CONF_MIN
            and best in co_seer):              # speaker MUST have CO'd Seer
        for tgt in div_by.get(best, {}).get("wolf", []):
            if tgt not in ws:
                continue
            if tgt in trusted_medium_human:
                continue  # Medium says human → reject this Seer call
            ws[tgt] = min(1.0, ws[tgt] + SEER_WOLF_NUDGE * best_conf)

    # Other divination reporters that are clearly low-confidence but still
    # claimed Seer-like → bump Madman (fake Seer is usually Madman).
    for reporter, c in confs.items():
        if reporter == best:
            continue
        if reporter in co_seer and c < 0.4:
            rs[reporter]["Madman"] = rs[reporter].get("Madman", 0.0) + CHAIN_SCALE * 0.25
            rs[reporter]["Seer"] = max(0.0, rs[reporter].get("Seer", 0.0) - CHAIN_SCALE * 0.10)

    return ws, rs
