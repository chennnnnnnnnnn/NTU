"""Constrained role solver.

Given per-player wolf_score, role_scores, and the role_counts parsed from
the game header, output a legal role assignment.

Assignment priority: Werewolf -> Seer -> Medium -> Madman -> Hunter -> Villager.

Tie-break for Werewolf candidates (descending):
    (wolf_score, role_scores["Werewolf"], -role_scores["Madman"], -role_scores["Villager"])

Tie-break for special roles X (descending):
    (role_scores[X], -wolf_score, -role_scores["Werewolf"])
"""
from __future__ import annotations

from typing import Dict, List, Tuple

ALL_ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Hunter", "Madman"]
ASSIGN_ORDER = ["Werewolf", "Seer", "Medium", "Madman", "Hunter"]


def _wolf_sort_key(p: str, wolf_score: Dict[str, float], role_scores: Dict[str, Dict[str, float]]):
    rs = role_scores.get(p, {})
    # Deprioritize players who staked a claim on any non-Werewolf special role
    # (CO Seer / Medium / Hunter, including high-Madman from contradiction-detection).
    # Without this, a fake Seer (true Madman) with execution-bonus wolf_score
    # can tie with real wolves and steal a Werewolf slot.
    special_claim = sum(rs.get(r, 0.0) for r in ("Seer", "Medium", "Hunter", "Madman"))
    return (
        wolf_score.get(p, 0.0),
        rs.get("Werewolf", 0.0),
        -special_claim,
        -rs.get("Villager", 0.0),
        p,
    )


def _special_sort_key(p: str, role: str, wolf_score: Dict[str, float], role_scores: Dict[str, Dict[str, float]]):
    rs = role_scores.get(p, {})
    return (
        rs.get(role, 0.0),
        -wolf_score.get(p, 0.0),
        -rs.get("Werewolf", 0.0),
        p,
    )


def assign(
    players: List[str],
    role_counts: Dict[str, int],
    wolf_score: Dict[str, float],
    role_scores: Dict[str, Dict[str, float]],
    order: List[str] | None = None,
) -> Dict[str, str]:
    """Return mapping player -> role.

    `order` overrides ASSIGN_ORDER (default Werewolf-first). Werewolf uses the
    wolf sort key; all other roles use the special sort key. Villager always
    last. wolf_score COLUMN is unaffected (only the role column changes).
    """
    seq = order if order is not None else ASSIGN_ORDER
    remaining = list(dict.fromkeys(players))
    assignment: Dict[str, str] = {}

    for role in seq:
        if role == "Villager":
            continue
        k = role_counts.get(role, 0)
        if k <= 0:
            continue
        if role == "Werewolf":
            key = lambda p: _wolf_sort_key(p, wolf_score, role_scores)
        else:
            key = lambda p, _r=role: _special_sort_key(p, _r, wolf_score, role_scores)
        candidates = sorted(remaining, key=key, reverse=True)
        for p in candidates[:k]:
            assignment[p] = role
        remaining = [p for p in remaining if p not in assignment]

    for p in remaining:
        assignment[p] = "Villager"
    return assignment


def cap_role_counts(role_counts: Dict[str, int], n_players: int) -> Dict[str, int]:
    """Adjust role_counts so they sum to exactly n_players.

    For the partial-prediction edge case (e.g. private/08 predicts 3 of 15
    players) the parsed role_counts overshoot.  We scale down each non-Villager
    role proportionally and put the remainder as Villager.
    """
    total = sum(role_counts.values())
    if total == n_players:
        return dict(role_counts)
    if total == 0:
        return {"Villager": n_players}

    # Cap each role to at most n_players; trim from Villager first, then evenly.
    capped = {r: min(c, n_players) for r, c in role_counts.items()}
    extra = sum(capped.values()) - n_players
    if extra > 0:
        # Drop Villager count first.
        v = capped.get("Villager", 0)
        drop = min(v, extra)
        capped["Villager"] = v - drop
        extra -= drop
    # Then drop minority roles (Madman, Hunter, Medium, Seer, Werewolf) in that order.
    for role in ["Madman", "Hunter", "Medium", "Seer", "Werewolf"]:
        if extra <= 0:
            break
        c = capped.get(role, 0)
        drop = min(c, extra)
        capped[role] = c - drop
        extra -= drop

    # Pad with Villager if we under-shot (shouldn't happen but be safe).
    diff = n_players - sum(capped.values())
    if diff > 0:
        capped["Villager"] = capped.get("Villager", 0) + diff

    return capped


def fallback_role_counts(n_players: int) -> Dict[str, int]:
    """Use Dataset_README rules when header parse failed."""
    if n_players >= 13:
        wolves = 3
    elif n_players >= 10:
        wolves = 2
    else:
        wolves = max(1, n_players // 5)
    seer = 1 if n_players >= 10 else 0
    medium = 1 if n_players >= 10 else 0
    hunter = 1 if n_players >= 11 else 0
    madman = 1 if n_players >= 11 else 0
    villager = n_players - (wolves + seer + medium + hunter + madman)
    villager = max(0, villager)
    return {
        "Villager": villager,
        "Werewolf": wolves,
        "Seer": seer,
        "Medium": medium,
        "Hunter": hunter,
        "Madman": madman,
    }
