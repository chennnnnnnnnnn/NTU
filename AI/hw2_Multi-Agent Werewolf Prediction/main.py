"""Main pipeline.

v1 = rule-only (night-death=0, exec+0.15, single-Seer-black+0.20).
v2 = v1 + post-based evidence (CO Seer/Medium/Hunter, divine results,
     claim conflict, not-Seer-not-Medium disclaim).

Usage:
    python main.py \
        --data-dir /home/htiintern2502/AI2/data/Werewolf_Prediction_Dataset \
        --split public \
        --output artifacts/submissions/v2_public.csv \
        --version v2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'src'))

from evidence_extractor import extract_evidence
from parsers.names import build_speaker_map
from parsers.posts import parse_posts
from parsers.role_counts import parse_role_counts
from solver import (
    ALL_ROLES,
    assign,
    cap_role_counts,
    fallback_role_counts,
)

# --- baseline rules ---
WOLF_PRIOR = 0.15
EXEC_BONUS = 0.15
SEER_BLACK_BONUS_V1 = 0.20

# --- v2 rules ---
SINGLE_SEER_CO_BOOST = 0.50
MULTI_SEER_CO_SEER_BOOST = 0.30   # ↑ at least one multi-CO claimer IS the real Seer
MULTI_SEER_CO_MADMAN_BOOST = 0.25  # fakes are usually Madman (human), not wolf
MULTI_SEER_CO_WEREWOLF_BOOST = 0.05  # ↓ rarely a true wolf
MULTI_CO_WOLF_NUDGE = -0.05  # ↓↓ NEGATIVE: claimed Seers should NOT enter wolf top-K
                              # (real Seer is never wolf; fakes are mostly Madman)

SINGLE_MED_CO_BOOST = 0.50
MULTI_MED_CO_MED_BOOST = 0.30   # at least one multi-CO claimer IS the real Medium
MULTI_MED_CO_MADMAN_BOOST = 0.25  # fakes are usually Madman (human)
MULTI_MED_CO_WEREWOLF_BOOST = 0.05  # rarely a true wolf

HUNTER_CO_BOOST_SINGLE = 0.40
HUNTER_CO_BOOST_MULTI = 0.15

DIVINE_BLACK_WOLF_BONUS = 0.30
DIVINE_WHITE_WOLF_REDUCE = 0.08
DIVINE_BLACK_ROLE_BONUS = 0.20

# --- v3 Madman reverse-detection ---
# Strategy: contradicted CO Seer/Medium gets BIG Madman bump but small
# wolf_score bump — we don't want them to take a Werewolf slot from real wolves.
FAKE_MADMAN_BASE = 0.40
FAKE_MADMAN_PER_HIT = 0.12
FAKE_WEREWOLF_BASE = 0.05
FAKE_WEREWOLF_PER_HIT = 0.03
FAKE_WOLF_SCORE_BASE = 0.0     # no wolf_score nudge — keeps Madman out of wolf top-k
FAKE_WOLF_SCORE_PER_HIT = 0.02
FAKE_SEER_PENALTY = 0.20
CONTRADICTION_CAP = 3

ROLE_PRIOR = {
    "Villager": 0.50,
    "Werewolf": 0.20,
    "Seer":     0.08,
    "Medium":   0.08,
    "Hunter":   0.07,
    "Madman":   0.07,
}

# --- regex (used by v1 path) ---
NIGHT_DEATH_RE = re.compile(r"The next morning, (.+?) was found in a gruesome state\.")
EXECUTION_RE = re.compile(r"^(.+?) was executed by the villagers\.")
SEER_CO_RE_V1 = re.compile(r"\[Seer\]:|I am the Seer|I'm the Seer", re.IGNORECASE)


# -------------------- v1 path --------------------
def extract_events_v1(log_text: str, players: list[str]) -> dict:
    player_set = set(players)
    night_deaths: list[str] = []
    executions: list[str] = []
    for line in log_text.splitlines():
        line_str = line.strip()
        m = NIGHT_DEATH_RE.search(line_str)
        if m and m.group(1) in player_set:
            night_deaths.append(m.group(1))
            continue
        m = EXECUTION_RE.match(line_str)
        if m and m.group(1) in player_set:
            executions.append(m.group(1))

    lines = log_text.splitlines()
    seer_claims: set[str] = set()
    for i, line in enumerate(lines):
        if SEER_CO_RE_V1.search(line):
            for j in range(i - 1, max(-1, i - 9), -1):
                s = lines[j].strip()
                if s in player_set:
                    seer_claims.add(s)
                    break

    seer_blacks: set[str] = set()
    if len(seer_claims) == 1:
        for p in player_set:
            pe = re.escape(p)
            patterns = [
                rf"\b{pe}\s+is\s+(?:a\s+)?(?:wolf|werewolf)\b",
                rf"\b{pe}\s+is\s+black\b",
                rf"\[{pe}\s+is\s+(?:a\s+)?(?:wolf|werewolf)\b",
            ]
            for pat in patterns:
                if re.search(pat, log_text, re.IGNORECASE):
                    seer_blacks.add(p)
                    break

    return {
        "night_deaths": night_deaths,
        "executions": executions,
        "seer_claims": seer_claims,
        "seer_blacks": seer_blacks,
    }


def compute_v1_scores(players: list[str], events: dict) -> tuple[dict, dict]:
    wolf_score = {p: WOLF_PRIOR for p in players}
    for victim in events["night_deaths"]:
        wolf_score[victim] = 0.0
    for executed in events["executions"]:
        wolf_score[executed] = wolf_score.get(executed, WOLF_PRIOR) + EXEC_BONUS
    if len(events["seer_claims"]) == 1:
        for target in events["seer_blacks"]:
            wolf_score[target] = wolf_score.get(target, WOLF_PRIOR) + SEER_BLACK_BONUS_V1
    wolf_score = {p: max(0.0, min(1.0, s)) for p, s in wolf_score.items()}
    role_scores = {p: dict(ROLE_PRIOR) for p in players}
    return wolf_score, role_scores


# -------------------- v2 path --------------------
def _apply_co(role_scores: dict, wolf_score: dict, player: str, role: str, boost: float):
    if player not in role_scores:
        return
    role_scores[player][role] = role_scores[player].get(role, 0.0) + boost


def _compute_fake_seer_scores(evidence: dict) -> dict:
    """Count contradictions for each claimed Seer / Medium.

    Sources of contradiction:
      - their black target is a night-death (wolves don't get killed at night) ×2
      - their white target is the trusted Medium's black ×2
      - their black target is the trusted Medium's white ×2
      - cross-Seer disagreement on the same target ×1
    "Trusted Medium" is the single CO Medium if exactly one Medium claimed.
    """
    night_deaths_set = set(evidence["night_deaths"])

    trusted_wolves: set = set()
    trusted_humans: set = set()
    if len(evidence["co_medium"]) == 1:
        med = next(iter(evidence["co_medium"]))
        mb = evidence["medium_blacks"].get(med, set())
        mw = evidence["medium_whites"].get(med, set())
        trusted_wolves = mb - mw
        trusted_humans = mw - mb

    fake_seer: dict = {}
    for s in evidence["co_seer"]:
        b = evidence["seer_blacks"].get(s, set())
        w = evidence["seer_whites"].get(s, set())
        score = 0
        score += 2 * len(b & night_deaths_set)
        score += 2 * len(b & trusted_humans)
        score += 2 * len(w & trusted_wolves)
        for other in evidence["co_seer"]:
            if other == s:
                continue
            ob = evidence["seer_blacks"].get(other, set())
            ow = evidence["seer_whites"].get(other, set())
            score += len(b & ow) + len(w & ob)
        fake_seer[s] = score

    # Symmetric: also compute fake-medium score (a fake Medium contradicts itself)
    fake_med: dict = {}
    for m in evidence["co_medium"]:
        b = evidence["medium_blacks"].get(m, set())
        w = evidence["medium_whites"].get(m, set())
        score = 0
        score += 2 * len(b & night_deaths_set)
        # Cross-medium conflict
        for other in evidence["co_medium"]:
            if other == m:
                continue
            ob = evidence["medium_blacks"].get(other, set())
            ow = evidence["medium_whites"].get(other, set())
            score += len(b & ow) + len(w & ob)
        fake_med[m] = score

    return {"seer": fake_seer, "medium": fake_med}


def compute_v2_scores(players: list[str], evidence: dict) -> tuple[dict, dict]:
    """Apply v1 rules + structured evidence from evidence_extractor."""
    wolf_score = {p: WOLF_PRIOR for p in players}
    role_scores = {p: dict(ROLE_PRIOR) for p in players}

    # --- v1 carry-over ---
    for victim in evidence["night_deaths"]:
        if victim in wolf_score:
            wolf_score[victim] = 0.0
            role_scores[victim]["Werewolf"] = 0.0  # certainly not wolf

    for executed in evidence["executions"]:
        if executed in wolf_score:
            wolf_score[executed] = wolf_score.get(executed, WOLF_PRIOR) + EXEC_BONUS

    fake = _compute_fake_seer_scores(evidence)

    # --- CO Seer (v3: contradiction-aware) ---
    n_seer = len(evidence["co_seer"])
    for p in evidence["co_seer"]:
        f = fake["seer"].get(p, 0)
        hits = min(f, CONTRADICTION_CAP)
        if f == 0:
            # No detected contradiction → trust as Seer
            if n_seer == 1:
                _apply_co(role_scores, wolf_score, p, "Seer", SINGLE_SEER_CO_BOOST)
            else:
                _apply_co(role_scores, wolf_score, p, "Seer", MULTI_SEER_CO_SEER_BOOST)
                _apply_co(role_scores, wolf_score, p, "Madman", MULTI_SEER_CO_MADMAN_BOOST)
                _apply_co(role_scores, wolf_score, p, "Werewolf", MULTI_SEER_CO_WEREWOLF_BOOST)
                if p in wolf_score:
                    wolf_score[p] += MULTI_CO_WOLF_NUDGE
        else:
            # Contradicted → fake Seer. wolf nudge ALWAYS (preserves v7 AP lock);
            # role boost/penalty gated by vB3 flag.
            if p in wolf_score:
                wolf_score[p] += FAKE_WOLF_SCORE_BASE + FAKE_WOLF_SCORE_PER_HIT * hits
            if not DISABLE_FAKE_ROLE:
                _apply_co(role_scores, wolf_score, p, "Madman",
                          FAKE_MADMAN_BASE + FAKE_MADMAN_PER_HIT * hits)
                _apply_co(role_scores, wolf_score, p, "Werewolf",
                          FAKE_WEREWOLF_BASE + FAKE_WEREWOLF_PER_HIT * hits)
                if p in role_scores:
                    role_scores[p]["Seer"] = max(0.0, role_scores[p]["Seer"] - FAKE_SEER_PENALTY)

    # --- CO Medium (v3: contradiction-aware) ---
    n_med = len(evidence["co_medium"])
    for p in evidence["co_medium"]:
        f = fake["medium"].get(p, 0)
        hits = min(f, CONTRADICTION_CAP)
        if f == 0:
            if n_med == 1:
                _apply_co(role_scores, wolf_score, p, "Medium", SINGLE_MED_CO_BOOST)
            else:
                _apply_co(role_scores, wolf_score, p, "Medium", MULTI_MED_CO_MED_BOOST)
                _apply_co(role_scores, wolf_score, p, "Madman", MULTI_MED_CO_MADMAN_BOOST)
                _apply_co(role_scores, wolf_score, p, "Werewolf", MULTI_MED_CO_WEREWOLF_BOOST)
                if p in wolf_score:
                    wolf_score[p] += MULTI_CO_WOLF_NUDGE
        else:
            if p in wolf_score:
                wolf_score[p] += FAKE_WOLF_SCORE_BASE + FAKE_WOLF_SCORE_PER_HIT * hits
            if not DISABLE_FAKE_ROLE:
                _apply_co(role_scores, wolf_score, p, "Madman",
                          FAKE_MADMAN_BASE + FAKE_MADMAN_PER_HIT * hits)
                _apply_co(role_scores, wolf_score, p, "Werewolf",
                          FAKE_WEREWOLF_BASE + FAKE_WEREWOLF_PER_HIT * hits)
                if p in role_scores:
                    role_scores[p]["Medium"] = max(0.0, role_scores[p]["Medium"] - FAKE_SEER_PENALTY)

    # --- CO Hunter ---
    n_hunter = len(evidence["co_hunter"])
    for p in evidence["co_hunter"]:
        boost = HUNTER_CO_BOOST_SINGLE if n_hunter == 1 else HUNTER_CO_BOOST_MULTI
        _apply_co(role_scores, wolf_score, p, "Hunter", boost)

    # --- not-Seer-not-Medium disclaim: zero out Seer/Medium odds (R9) ---
    if not DISABLE_DISCLAIM_ZERO:
        for p in evidence["not_seer_medium"]:
            if p in role_scores:
                role_scores[p]["Seer"] = 0.0
                role_scores[p]["Medium"] = 0.0

    # --- Divine results: trust single CO Seer ---
    if n_seer == 1:
        seer = next(iter(evidence["co_seer"]))
        blacks = evidence["seer_blacks"].get(seer, set())
        whites = evidence["seer_whites"].get(seer, set())
        clean_blacks = blacks - whites
        clean_whites = whites - blacks
        for target in clean_blacks:
            if target in wolf_score:
                wolf_score[target] += DIVINE_BLACK_WOLF_BONUS
                role_scores[target]["Werewolf"] += DIVINE_BLACK_ROLE_BONUS
        for target in clean_whites:
            if target in wolf_score:
                wolf_score[target] -= DIVINE_WHITE_WOLF_REDUCE

    # --- Medium results: trust single CO Medium ---
    if n_med == 1:
        med = next(iter(evidence["co_medium"]))
        blacks = evidence["medium_blacks"].get(med, set())
        whites = evidence["medium_whites"].get(med, set())
        clean_blacks = blacks - whites
        clean_whites = whites - blacks
        for target in clean_blacks:
            if target in wolf_score:
                wolf_score[target] += DIVINE_BLACK_WOLF_BONUS
                role_scores[target]["Werewolf"] += DIVINE_BLACK_ROLE_BONUS
        for target in clean_whites:
            if target in wolf_score:
                wolf_score[target] -= DIVINE_WHITE_WOLF_REDUCE

    # Clip
    wolf_score = {p: max(0.0, min(1.0, s)) for p, s in wolf_score.items()}
    return wolf_score, role_scores


# -------------------- v3+LLM ensemble --------------------
# v4 setup: LLM_WEIGHT 0.3 for wolf_score, LLM_ROLE_WEIGHT 0 for role.
# v5 setup (RAG): three-way blend with rank-based calibration.
LLM_WEIGHT = 0.4         # weight on LLM raw wolf_score
LLM_ROLE_WEIGHT = 0.0    # weight on LLM role_scores
LLM_RANK_WEIGHT = 0.0    # weight on rank-derived score (0 disables; v5 uses 0.10)
# v8 setup: Qwen-14B reviewer nudge on top wolf candidates only.
REVIEWER_WEIGHT = 0.15
SOLVER_ORDER = None
DISABLE_DISCLAIM_ZERO = False
DISABLE_FAKE_ROLE = False   # weight on reviewer_wolf_score (only applied to reviewed players)


def _rank_score_from_preds(players: list[str], llm_preds: dict) -> dict:
    """Convert LLM `rank` field (or wolf_score if missing) into a [0, 1] score.

    Rank 0 (top wolf candidate) → 1.0; rank N-1 → 0.0.
    Falls back to ranking by llm wolf_score if LLM didn't return wolf_ranking.
    """
    if not llm_preds:
        return {p: 0.0 for p in players}
    ranks = {p: llm_preds[p].get("rank") for p in players if p in llm_preds}
    if not any(r is not None for r in ranks.values()):
        # Derive rank from wolf_score
        scored = sorted(
            [(p, llm_preds[p]["wolf_score"]) for p in players if p in llm_preds],
            key=lambda x: -x[1],
        )
        ranks = {p: i for i, (p, _) in enumerate(scored)}
    N = max(2, len(ranks))
    out = {}
    for p in players:
        r = ranks.get(p)
        if r is None:
            out[p] = 0.0
        else:
            out[p] = max(0.0, 1.0 - r / (N - 1))
    return out


def apply_reviewer(
    players: list[str],
    wolf_score: dict,
    role_scores: dict,
    reviews: dict,
    weight: float = None,
) -> tuple[dict, dict]:
    """Blend the Qwen-14B reviewer's per-candidate scores into the post-v7 wolf_score.

    For each reviewed player:
        final_wolf = (1 - weight) * v7_wolf + weight * reviewer_wolf_score
    Non-reviewed players are unchanged.

    Also: when madman_likelihood is high (>= 0.5) AND decision is 'demote',
    bump role_scores[Madman] for that player by 0.20 (small, doesn't override
    CO-confirmed Seer/Medium/Hunter slots).
    """
    if not reviews:
        return wolf_score, role_scores
    w = REVIEWER_WEIGHT if weight is None else weight
    out_wolf = dict(wolf_score)
    out_role = {p: dict(rs) for p, rs in role_scores.items()}
    for p in players:
        review = reviews.get(p)
        if review is None:
            continue
        rev_ws = review["reviewer_wolf_score"]
        out_wolf[p] = (1 - w) * wolf_score.get(p, 0.15) + w * rev_ws
        mad = review.get("madman_likelihood", 0.0)
        if review["decision"] == "demote" and mad >= 0.5:
            out_role[p]["Madman"] = out_role[p].get("Madman", 0.0) + 0.20
    out_wolf = {p: max(0.0, min(1.0, s)) for p, s in out_wolf.items()}
    return out_wolf, out_role


def ensemble_v2_llm(
    players: list[str],
    rule_wolf: dict, rule_role: dict,
    llm_preds: dict,
    role_weight: float = None,
) -> tuple[dict, dict]:
    """Blend rule-based scores with LLM predictions.

    wolf_score:  w_rule * rule + w_llm * llm + w_rank * rank_score
                  where w_rule = 1 - w_llm - w_rank
    role_scores: (1 - w_role) * rule + w_role * llm   (LLM_ROLE_WEIGHT;
                 default 0 keeps rule's CO-detected Seer/Medium signals.)
    Players missing from LLM output → keep rule scores entirely.
    """
    w_llm = LLM_WEIGHT
    w_rank = LLM_RANK_WEIGHT
    w_rule = max(0.0, 1.0 - w_llm - w_rank)
    w_role = LLM_WEIGHT if role_weight is None else role_weight

    rank_scores = _rank_score_from_preds(players, llm_preds) if llm_preds else {}

    out_wolf = {}
    out_role = {}
    for p in players:
        rw = rule_wolf.get(p, 0.15)
        rr = rule_role.get(p, dict(ROLE_PRIOR))
        pred = llm_preds.get(p) if llm_preds else None
        if pred is None:
            out_wolf[p] = rw
            out_role[p] = dict(rr)
            continue
        lw = pred["wolf_score"]
        rs = rank_scores.get(p, 0.0)
        out_wolf[p] = w_rule * rw + w_llm * lw + w_rank * rs
        merged = {}
        for r in ROLE_PRIOR:
            merged[r] = (1 - w_role) * rr.get(r, 0.0) + w_role * pred["role_scores"].get(r, 0.0)
        out_role[p] = merged
    out_wolf = {p: max(0.0, min(1.0, s)) for p, s in out_wolf.items()}
    return out_wolf, out_role


# -------------------- pipeline glue --------------------
def process_game(
    game_idx: str,
    sub_df: pd.DataFrame,
    log_path: Path,
    version: str,
    llm_preds_by_game: dict | None = None,
    reviewer_preds_by_game: dict | None = None,
    results_dir: Path | None = None,
    medium_chain_dir: Path | None = None,
    seer_chain_enabled: bool = False,
    hunter_chain_enabled: bool = False,
    madman_repair_enabled: bool = False,
    split: str | None = None,
) -> pd.DataFrame:
    players = sub_df["character"].tolist()
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    role_counts = parse_role_counts(log_text)
    if not role_counts:
        role_counts = fallback_role_counts(len(players))
    role_counts = cap_role_counts(role_counts, len(players))

    if version == "v1":
        events = extract_events_v1(log_text, players)
        wolf_score, role_scores = compute_v1_scores(players, events)
    else:  # v2 / v2+llm
        posts = parse_posts(log_text, players)
        speaker_map = build_speaker_map(players, {p.speaker for p in posts})
        evidence = extract_evidence(posts, players, speaker_map, log_text)
        wolf_score, role_scores = compute_v2_scores(players, evidence)

    if version in {"v2+llm", "v2+llm+rev"} and llm_preds_by_game is not None:
        llm_preds = llm_preds_by_game.get(game_idx)
        wolf_score, role_scores = ensemble_v2_llm(
            players, wolf_score, role_scores, llm_preds, role_weight=LLM_ROLE_WEIGHT,
        )

    if version == "v2+llm+rev" and reviewer_preds_by_game is not None:
        reviews = reviewer_preds_by_game.get(game_idx)
        wolf_score, role_scores = apply_reviewer(players, wolf_score, role_scores, reviews)

    # Step 1: LLM-extracted RESULT evidence → Seer/Medium role + trusted wolf boost
    if results_dir is not None and split is not None:
        from results_scorer import load_and_score
        ev_basic = extract_events_v1(log_text, players)
        rp = results_dir / f"{split}_{game_idx}.json"
        role_delta, wolf_delta = load_and_score(
            rp, ev_basic["executions"], ev_basic["night_deaths"]
        )
        for p, dd in role_delta.items():
            if p in role_scores:
                for r, v in dd.items():
                    role_scores[p][r] = role_scores[p].get(r, 0.0) + v
        for tgt, dv in wolf_delta.items():
            if tgt in wolf_score:
                wolf_score[tgt] = max(0.0, min(1.0, wolf_score[tgt] + dv))

    # Medium-execution chain + Seer-divination chain (rule-based, ROLE-ONLY).
    if medium_chain_dir is not None and split is not None:
        import medium_chain as _mcmod
        _mcmod.VC7_GAME = f'{split}/{game_idx}'
        from medium_chain import apply_medium_chain, load_results, medium_confidence
        from seer_chain import apply_seer_chain
        results = load_results(medium_chain_dir, split, game_idx)
        if results:
            posts2 = parse_posts(log_text, players)
            spk2 = build_speaker_map(players, {p.speaker for p in posts2})
            ev2 = extract_evidence(posts2, players, spk2, log_text)
            ev_basic2 = extract_events_v1(log_text, players)
            co_med = ev2.get("co_medium", set())
            not_sm = ev2.get("not_seer_medium", set())
            execs = ev_basic2["executions"]
            ndeaths = ev_basic2["night_deaths"]
            wolf_score, role_scores = apply_medium_chain(
                players, wolf_score, role_scores, results,
                execs, ndeaths, co_med, not_sm,
            )
            # Derive trusted-Medium reveals to cross-check Seer chain
            tm_wolf, tm_human = set(), set()
            med_by = results.get("medium_by", {}) or {}
            if med_by:
                best_m, best_c = None, 0.0
                for rep, r in med_by.items():
                    c = medium_confidence(rep, r, set(execs), set(ndeaths),
                                          set(co_med), set(not_sm))
                    if c > best_c:
                        best_c, best_m = c, rep
                if best_m and best_c >= 0.5:
                    tm_wolf = set(med_by[best_m].get("wolf", []))
                    tm_human = set(med_by[best_m].get("human", []))
            if seer_chain_enabled:
                wolf_score, role_scores = apply_seer_chain(
                    players, wolf_score, role_scores, results,
                    ev2.get("co_seer", set()), not_sm, tm_wolf, tm_human,
                )
    if hunter_chain_enabled and split is not None:
        from hunter_chain import apply_hunter_chain
        wolf_score, role_scores = apply_hunter_chain(
            players, wolf_score, role_scores, log_text,
        )
    if madman_repair_enabled and medium_chain_dir is not None and split is not None:
        from madman_repair import apply_madman_repair
        from medium_chain import load_results as _lr
        _res = _lr(medium_chain_dir, split, game_idx)
        _posts = parse_posts(log_text, players)
        _spk = build_speaker_map(players, {p.speaker for p in _posts})
        _ev = extract_evidence(_posts, players, _spk, log_text)
        _fake = _compute_fake_seer_scores(_ev)
        wolf_score, role_scores = apply_madman_repair(
            players, wolf_score, role_scores, role_counts, _fake,
            _ev.get("co_seer", set()), _ev.get("co_medium", set()), _res,
        )

    role_assign = assign(players, role_counts, wolf_score, role_scores, order=SOLVER_ORDER)

    out = sub_df.copy()
    out["role"] = out["character"].map(role_assign)
    out["wolf_score"] = out["character"].map(wolf_score)
    return out


def _load_llm_predictions(llm_dir: Path | None, split: str) -> dict | None:
    """Load pre-computed per-game LLM predictions: {game_idx: {player: {...}}}."""
    if llm_dir is None:
        return None
    out = {}
    for f in sorted(llm_dir.glob(f"{split}_*.predictions.json")):
        # filename: public_01.predictions.json
        game_idx = f.stem.replace(f"{split}_", "").replace(".predictions", "")
        try:
            out[game_idx] = json.loads(f.read_text())
        except Exception as e:
            print(f"[warn] failed to read {f}: {e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=["public", "private"])
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--version", choices=["v1", "v2", "v2+llm", "v2+llm+rev"], default="v2")
    ap.add_argument("--llm-dir", type=Path, default=None,
                    help="Dir with {split}_{game}.predictions.json from run_llm.py")
    ap.add_argument("--reviewer-dir", type=Path, default=None,
                    help="Dir with {split}_{game}.reviews.json from run_reviewer.py")
    ap.add_argument("--reviewer-weight", type=float, default=None,
                    help="Override reviewer ensemble weight (default 0.15)")
    ap.add_argument("--llm-weight", type=float, default=None,
                    help="Override ensemble weight on LLM wolf_score (default 0.4)")
    ap.add_argument("--llm-role-weight", type=float, default=None,
                    help="Override ensemble weight on LLM role_scores (default 0)")
    ap.add_argument("--llm-rank-weight", type=float, default=None,
                    help="Override weight on LLM rank-derived score (default 0)")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Dir with {split}_{game}.json from run_results_extract.py (Step 1)")
    ap.add_argument("--medium-chain-dir", type=Path, default=None,
                    help="Dir with {split}_{game}.json results for medium-execution chain")
    ap.add_argument("--seer-chain", action="store_true",
                    help="Also apply Seer-divination chain (role-only)")
    ap.add_argument("--hunter-chain", action="store_true",
                    help="Also apply Hunter-claim protection chain (role-only)")
    ap.add_argument("--madman-repair", action="store_true",
                    help="Apply Madman slot repair (role-only)")
    ap.add_argument("--solver-order", type=str, default=None,
                    help="Comma-separated role order")
    ap.add_argument("--chain-scale", type=float, default=1.0,
                    help="Soft weight on medium/seer chain role deltas (B1=0.10)")
    ap.add_argument("--no-multico-fix", action="store_true",
                    help="Revert multi-CO constants to original v7 values")
    ap.add_argument("--no-disclaim-zero", action="store_true",
                    help="vB2: skip R9 not-seer-not-medium hard zeroing (role-only)")
    ap.add_argument("--no-fake-role", action="store_true",
                    help="vB3: skip R3/R4 ...")
    ap.add_argument("--medium-wolf-nudge", type=float, default=0.0,
                    help="vC1: soft wolf nudge from high-conf Medium on executed targets")
    ap.add_argument("--medium-wolf-conf-min", type=float, default=0.8,
                    help="vC1: min medium_confidence to fire wolf nudge")
    ap.add_argument("--medium-wolf-cap", type=float, default=0.0,
                    help="vC1-cap: set ws=max(ws,CAP)")
    ap.add_argument("--seer-wolf-nudge", type=float, default=0.0,
                    help="vC2: soft wolf nudge from trusted Seer divination")
    ap.add_argument("--seer-wolf-conf-min", type=float, default=0.90,
                    help="vC2: min seer_confidence to fire (strict)")
    ap.add_argument("--medium-human-decay", type=float, default=0.0,
                    help="vC4: ws -= decay*conf ...")
    ap.add_argument("--vc7-single", type=float, default=0.0,
                    help="vC7: force single whitelist fire private/03 Friedel->Moritz at this nudge")
    args = ap.parse_args()
    if args.llm_weight is not None:
        global LLM_WEIGHT
        LLM_WEIGHT = args.llm_weight
    if args.llm_role_weight is not None:
        global LLM_ROLE_WEIGHT
        LLM_ROLE_WEIGHT = args.llm_role_weight
    if args.llm_rank_weight is not None:
        global LLM_RANK_WEIGHT
        LLM_RANK_WEIGHT = args.llm_rank_weight
    if args.reviewer_weight is not None:
        global REVIEWER_WEIGHT
        REVIEWER_WEIGHT = args.reviewer_weight
    global SOLVER_ORDER, DISABLE_DISCLAIM_ZERO, DISABLE_FAKE_ROLE
    SOLVER_ORDER = args.solver_order.split(',') if args.solver_order else None
    DISABLE_DISCLAIM_ZERO = args.no_disclaim_zero
    DISABLE_FAKE_ROLE = args.no_fake_role
    import medium_chain as _mc, seer_chain as _sc
    _mc.CHAIN_SCALE = args.chain_scale
    _sc.CHAIN_SCALE = args.chain_scale
    _mc.WOLF_SOFT_NUDGE = args.medium_wolf_nudge
    _mc.WOLF_SOFT_CONF_MIN = args.medium_wolf_conf_min
    _mc.WOLF_SOFT_CAP = args.medium_wolf_cap
    _sc.SEER_WOLF_NUDGE = args.seer_wolf_nudge
    _sc.SEER_WOLF_CONF_MIN = args.seer_wolf_conf_min
    _mc.WOLF_HUMAN_DECAY = args.medium_human_decay
    if args.vc7_single > 0.0:
        _mc.VC7_WHITELIST = {"private/03": [("Old Man Moritz", args.vc7_single)]}
    if args.no_multico_fix:
        global MULTI_SEER_CO_SEER_BOOST, MULTI_SEER_CO_MADMAN_BOOST, MULTI_SEER_CO_WEREWOLF_BOOST
        global MULTI_CO_WOLF_NUDGE, MULTI_MED_CO_MED_BOOST, MULTI_MED_CO_MADMAN_BOOST, MULTI_MED_CO_WEREWOLF_BOOST
        MULTI_SEER_CO_SEER_BOOST = 0.20
        MULTI_SEER_CO_MADMAN_BOOST = 0.20
        MULTI_SEER_CO_WEREWOLF_BOOST = 0.10
        MULTI_CO_WOLF_NUDGE = 0.05
        MULTI_MED_CO_MED_BOOST = 0.20
        MULTI_MED_CO_MADMAN_BOOST = 0.20
        MULTI_MED_CO_WEREWOLF_BOOST = 0.10

    roles_csv = args.data_dir / args.split / "roles.csv"
    df = pd.read_csv(roles_csv, dtype={"index": str})
    df["index"] = df["index"].str.zfill(2)
    df["index_str"] = df["index"]

    needs_llm = args.version in {"v2+llm", "v2+llm+rev"}
    llm_preds_by_game = _load_llm_predictions(args.llm_dir, args.split) if needs_llm else None
    if needs_llm:
        n = len(llm_preds_by_game) if llm_preds_by_game else 0
        print(f"Loaded {n} LLM prediction files from {args.llm_dir}; ensemble weight = {LLM_WEIGHT}")

    reviewer_preds_by_game = None
    if args.version == "v2+llm+rev":
        reviewer_preds_by_game = {}
        for f in sorted(args.reviewer_dir.glob(f"{args.split}_*.reviews.json")):
            game_idx = f.stem.replace(f"{args.split}_", "").replace(".reviews", "")
            try:
                reviewer_preds_by_game[game_idx] = json.loads(f.read_text())
            except Exception as e:
                print(f"[warn] failed to read {f}: {e}")
        print(f"Loaded {len(reviewer_preds_by_game)} reviewer files; reviewer weight = {REVIEWER_WEIGHT}")

    out_frames = []
    for game_idx, sub in df.groupby("index_str", sort=True):
        log_path = args.data_dir / args.split / f"{game_idx}.txt"
        if not log_path.exists():
            print(f"[warn] missing log: {log_path}, skipping")
            continue
        out_frames.append(process_game(
            game_idx, sub.drop(columns=["index_str"]), log_path, args.version,
            llm_preds_by_game=llm_preds_by_game,
            reviewer_preds_by_game=reviewer_preds_by_game,
            results_dir=args.results_dir,
            medium_chain_dir=args.medium_chain_dir,
            seer_chain_enabled=args.seer_chain,
            hunter_chain_enabled=args.hunter_chain,
            madman_repair_enabled=args.madman_repair,
            split=args.split,
        ))

    out = pd.concat(out_frames, ignore_index=True)
    out = out.sort_values("id").reset_index(drop=True)
    out = out[["id", "index", "character", "role", "wolf_score"]]

    assert list(out.columns) == ["id", "index", "character", "role", "wolf_score"]
    assert out["role"].notna().all()
    assert out["wolf_score"].notna().all()
    assert out["wolf_score"].between(0, 1).all()
    assert out["role"].isin(ALL_ROLES).all()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output} (version={args.version})")


if __name__ == "__main__":
    main()
