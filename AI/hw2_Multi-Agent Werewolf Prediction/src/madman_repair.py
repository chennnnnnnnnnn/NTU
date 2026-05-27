"""Madman slot repair (rule-based, ROLE-ONLY, conservative).

medium/seer chains rescued Seer/Medium but Madman F1 dropped (0.11 → 0.05)
because real Seer/Medium claimers were correctly pulled out, leaving the
solver's Madman step with worse candidates.

We add a conservative Madman role-score boost (NEVER touches wolf_score) using
5 signals — none of which depend on raising wolf_score:

  1. fake CO Seer/Medium (contradiction score > 0)
  2. claimed Seer/Medium but a chain identified a *different* player as the
     trusted real one → this claimer is a fake (Madman-leaning)
  3. "wolfy but not a confirmed wolf": high wolf_score yet ranked just OUTSIDE
     the top-`n_wolf` slot (acts suspicious but isn't a true wolf ⇒ Madman)
  4. claimed special role but Seer/Medium chain gave them low confidence
  5. disclaimed nothing + made many divination-style statements that the
     chains rejected
"""
from __future__ import annotations

from typing import Dict, Tuple


def apply_madman_repair(
    players: list[str],
    wolf_score: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    role_counts: Dict[str, int],
    fake_scores: Dict[str, dict],
    co_seer: set,
    co_medium: set,
    results: Dict,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    n_wolf = role_counts.get("Werewolf", 0)
    n_madman = role_counts.get("Madman", 0)
    rs = {p: dict(d) for p, d in role_scores.items()}
    if n_madman <= 0:
        return wolf_score, rs

    # Signal 3: ranked just below the wolf slot but still genuinely suspicious.
    if n_wolf > 0:
        order = sorted(players, key=lambda p: -wolf_score.get(p, 0.0))
        wolf_slot = set(order[:n_wolf])
        for i, p in enumerate(order):
            if p in wolf_slot or p not in rs:
                continue
            if n_wolf <= i < n_wolf + 4:
                ws = wolf_score.get(p, 0.0)
                if ws >= 0.20:
                    rs[p]["Madman"] = rs[p].get("Madman", 0.0) + 0.30 + 0.6 * (ws - 0.20)

    # Signal 1: fake CO claimers (self-contradiction score > 0).
    for kind in ("seer", "medium"):
        for p, f in (fake_scores.get(kind, {}) or {}).items():
            if f and p in rs:
                rs[p]["Madman"] = rs[p].get("Madman", 0.0) + 0.20

    # Signals 2 & 4: claimed Seer/Medium but the chains assigned the role
    # elsewhere → this claimer is likely the fake (Madman more than wolf).
    div_by = (results or {}).get("divination_by", {}) or {}
    med_by = (results or {}).get("medium_by", {}) or {}
    for p in co_seer:
        if p in rs:
            seer_sc = rs[p].get("Seer", 0.0)
            # low Seer score despite claiming → chain rejected them
            if seer_sc < 0.20:
                rs[p]["Madman"] = rs[p].get("Madman", 0.0) + 0.18
    for p in co_medium:
        if p in rs:
            med_sc = rs[p].get("Medium", 0.0)
            if med_sc < 0.20:
                rs[p]["Madman"] = rs[p].get("Madman", 0.0) + 0.18

    # Signal 5: produced divination-style results but is NOT the trusted Seer
    # (their Seer score stayed low) → noisy fake → Madman-leaning.
    for p, rep in div_by.items():
        if p not in rs:
            continue
        n_res = len(rep.get("wolf", [])) + len(rep.get("human", []))
        if n_res >= 2 and rs[p].get("Seer", 0.0) < 0.15:
            rs[p]["Madman"] = rs[p].get("Madman", 0.0) + 0.12

    return wolf_score, rs
