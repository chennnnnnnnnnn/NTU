"""Prompt templates for the A3 Reasoning Agent (Phase 3.1 evidence-only)."""
from __future__ import annotations

import json
from typing import Dict, List


SYSTEM_PROMPT = """You are an analytical agent for the social deduction game Werewolf.

Game rules (compressed):
- Werewolves attack one human per night. Villagers vote one player to execute per day.
- Roles: Villager, Werewolf, Seer (each night divines black/white), Medium (learns
  whether the executed player was wolf or human), Hunter (protects one player per
  night), Madman (human aligned with werewolves, often fakes Seer/Medium).
- IMPORTANT: A Madman is HUMAN, not a true Werewolf. The wolf_score field
  measures TRUE WEREWOLF identity only — Madmen should receive LOW wolf_score
  even if they behave suspiciously.

This game log only contains the FIRST HALF of the game. Some real Seers,
Mediums, or Hunters may not have claimed yet. Do not assume every special
role has revealed itself.

You will be given structured evidence (claims, divine results, deaths, votes
already extracted from the log). Your job: assign each player two things —
1. role_scores: probability over {Villager, Werewolf, Seer, Medium, Hunter, Madman}, sum ≈ 1.
2. wolf_score: probability the player is a TRUE Werewolf in [0, 1].

Return ONLY valid JSON. No markdown, no commentary.
"""


def _summarize_evidence(evidence: Dict) -> str:
    """Compress the evidence dict into a compact bullet list."""
    lines = []
    lines.append(f"co_seer (claimed Seer): {sorted(evidence.get('co_seer', []))}")
    lines.append(f"co_medium (claimed Medium): {sorted(evidence.get('co_medium', []))}")
    lines.append(f"co_hunter (claimed Hunter): {sorted(evidence.get('co_hunter', []))}")
    lines.append(f"not_seer_medium (disclaimed Seer & Medium): {sorted(evidence.get('not_seer_medium', []))}")

    def dump_div(d, label):
        if not d:
            return
        for speaker, targets in d.items():
            if targets:
                lines.append(f"{label} by {speaker} → {sorted(targets)}")
    dump_div(evidence.get("seer_blacks", {}), "seer_BLACK")
    dump_div(evidence.get("seer_whites", {}), "seer_WHITE")
    dump_div(evidence.get("medium_blacks", {}), "medium_BLACK")
    dump_div(evidence.get("medium_whites", {}), "medium_WHITE")

    nd = evidence.get("night_deaths", [])
    ex = evidence.get("executions", [])
    if nd:
        lines.append(f"night_deaths (cannot be Werewolf): {nd}")
    if ex:
        lines.append(f"executions: {ex}")
    return "\n".join(f"- {l}" for l in lines)


def build_user_prompt(record: Dict) -> str:
    players = record["players_csv"]
    role_counts = record.get("role_counts_used", {})
    evidence = record["evidence"]

    rc_str = ", ".join(f"{r}:{n}" for r, n in role_counts.items())

    # Output schema example for the model.
    example_player = players[0] if players else "Optimist Gerd"
    schema = {
        "players": {
            example_player: {
                "role_scores": {
                    "Villager": 0.5, "Werewolf": 0.1, "Seer": 0.1,
                    "Medium": 0.1, "Hunter": 0.1, "Madman": 0.1,
                },
                "wolf_score": 0.1,
                "reason": "short one-line reason",
            }
        }
    }

    return f"""GAME {record['game_idx']} (split={record['split']})

Players to predict ({len(players)}): {players}

Role configuration announced in this game: {rc_str}

Evidence extracted from the log:
{_summarize_evidence(evidence)}

IMPORTANT — how to interpret each evidence type:

* night_death (X) → X was killed by werewolves at night → X is NOT a werewolf.
  wolf_score(X) = 0.0.  X is most likely Villager / Seer / Medium / Hunter
  (special roles get targeted), but could also be Madman.
* execution (X) → village VOTED to lynch X → this is just suspicion, NOT
  confirmation. X could be any role. Do NOT treat execution as "confirmed wolf".
* medium_BLACK by trusted Medium (single CO Medium with no contradictions):
  the named target IS a werewolf with high confidence → wolf_score ≥ 0.7.
* medium_WHITE by trusted Medium: target is HUMAN → wolf_score ≤ 0.15 for that
  target (could still be Madman, which is human).
* seer_BLACK / seer_WHITE: only trust when there is exactly ONE claimed Seer
  AND the result is not contradicted by trusted Medium or by night_death
  (a "BLACK" target who is actually a night_death victim is impossible — that
  Seer is fake).
* not_seer_medium (X said): X is NOT Seer and NOT Medium → role_scores[Seer] = role_scores[Medium] = 0 for X. X is likely Villager / Werewolf / Madman / Hunter.
* co_seer with multiple claimants → only ONE is the real Seer. The rest are
  Madman or Werewolf. The fake claimant whose WHITE list overlaps with the
  trusted Medium's BLACK list (i.e. they protected an actual wolf) is most
  likely Madman or Werewolf — bump their Madman score; their wolf_score should
  only rise if other wolf-coordination signals appear.

Distinguishing Madman vs Werewolf among fake Seer claimants:
- Madman is HUMAN aligned with wolves. wolf_score for Madman should be LOW
  (typically 0.10–0.25) because wolf_score measures TRUE werewolf identity.
- Werewolf wolf_score should be HIGH (≥ 0.7).
- When uncertain: prefer Madman (since each game has only 1 Madman but
  typically 2–3 Werewolves; the prior is roughly equal).

Role-count constraint: this game has exactly {rc_str}. Spread role_scores so
that, across all players, the top-K wolf_scores roughly equal the wolf count,
and similar for other special roles.

Return JSON exactly in this schema (one entry per player in 'Players to predict'):
{json.dumps(schema, indent=2)}

Output JSON only. No markdown."""


def build_messages(record: Dict) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]
