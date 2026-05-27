"""Turn extracted investigation RESULTS into role/wolf score deltas.

Logic:
  - A real Medium reports results about EXECUTED players. So the reporter whose
    medium results best overlap with the game's execution list is the trusted
    Medium → big Medium role boost; their 'wolf' calls are trusted → wolf boost.
  - A real Seer reports divination results spread over many targets. The most
    prolific consistent divination reporter (not contradicted by trusted
    Medium) → Seer role boost; their 'wolf' calls → wolf boost.
  - Players who only report 1-2 results (likely just repeating others) → no boost.

Outputs per game: {player: {"seer": d, "medium": d}}, {target: wolf_delta}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


SEER_BOOST = 0.45
MEDIUM_BOOST = 0.45
WOLF_BOOST_TRUSTED = 0.30
WOLF_REDUCE_WHITE = 0.10
MIN_RESULTS_FOR_TRUST = 2


def score_from_results(
    results: Dict,
    executions: list[str],
    night_deaths: list[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Return (role_delta, wolf_delta)."""
    role_delta: Dict[str, Dict[str, float]] = {}
    wolf_delta: Dict[str, float] = {}

    divby = results.get("divination_by", {})
    medby = results.get("medium_by", {})
    exec_set = set(executions)

    def _ensure(p):
        if p not in role_delta:
            role_delta[p] = {"Seer": 0.0, "Medium": 0.0}

    # --- Trusted Medium: reporter whose medium targets best match executions ---
    best_medium = None
    best_overlap = 0
    for reporter, res in medby.items():
        targets = set(res.get("wolf", [])) | set(res.get("human", []))
        if len(targets) < MIN_RESULTS_FOR_TRUST:
            continue
        overlap = len(targets & exec_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_medium = reporter

    if best_medium and best_overlap >= 1:
        _ensure(best_medium)
        role_delta[best_medium]["Medium"] += MEDIUM_BOOST
        # Trusted medium's wolf calls about executed players → strong wolf signal
        for tgt in medby[best_medium].get("wolf", []):
            if tgt in exec_set or tgt not in night_deaths:
                wolf_delta[tgt] = wolf_delta.get(tgt, 0.0) + WOLF_BOOST_TRUSTED
        for tgt in medby[best_medium].get("human", []):
            wolf_delta[tgt] = wolf_delta.get(tgt, 0.0) - WOLF_REDUCE_WHITE

    # --- Trusted Seer: most prolific divination reporter, not the medium ---
    best_seer = None
    best_count = 0
    for reporter, res in divby.items():
        if reporter == best_medium:
            continue
        cnt = len(set(res.get("wolf", [])) | set(res.get("human", [])))
        if cnt < MIN_RESULTS_FOR_TRUST:
            continue
        if cnt > best_count:
            best_count = cnt
            best_seer = reporter

    if best_seer:
        _ensure(best_seer)
        role_delta[best_seer]["Seer"] += SEER_BOOST
        # Seer wolf calls → moderate wolf signal (Seer can be fake; weight lower
        # than medium). Only if not contradicted by trusted medium's whites.
        med_white = set(medby.get(best_medium, {}).get("human", [])) if best_medium else set()
        for tgt in divby[best_seer].get("wolf", []):
            if tgt in med_white:
                continue  # contradicted → skip
            if tgt not in night_deaths:
                wolf_delta[tgt] = wolf_delta.get(tgt, 0.0) + 0.5 * WOLF_BOOST_TRUSTED

    return role_delta, wolf_delta


def load_and_score(
    results_path: Path,
    executions: list[str],
    night_deaths: list[str],
):
    if not results_path.exists():
        return {}, {}
    results = json.loads(results_path.read_text())
    return score_from_results(results, executions, night_deaths)
