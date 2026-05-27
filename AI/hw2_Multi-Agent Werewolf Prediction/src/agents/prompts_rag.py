"""Phase 3.2 prompt: evidence + RAG retrieved posts + ranking-first output.

Differences from agents/prompts.py:
    - Includes per-player retrieved posts.
    - Asks the model to FIRST produce a wolf_ranking (ordered list of all
      players) before per-player scores. AP cares about ranking, so making the
      model commit to a ranking first usually yields more separable scores.
    - Explicit Madman vs Werewolf distinction list.
"""
from __future__ import annotations

import json
from typing import Dict, List


SYSTEM_PROMPT_RAG = """You are an analytical agent for Werewolf, a social deduction game.

Rules summary:
- Roles: Villager, Werewolf, Seer, Medium, Hunter, Madman.
- Werewolves kill one player at night (night_death) — wolves never get
  killed at night, so any night_death victim is HUMAN.
- The Seer divines black (wolf) / white (human) each night.
- The Medium learns whether each executed player was wolf or human.
- The Madman is HUMAN aligned with wolves, often fakes Seer/Medium and tries
  to mislead the village.

CRITICAL:
- wolf_score measures TRUE Werewolf identity ONLY.
- Madmen behave like wolves but are HUMAN — their wolf_score should be LOW
  (typically 0.05–0.25), even if they look suspicious.
- The log only covers the FIRST HALF of the game. Some real Seers/Mediums/
  Hunters may not have CO'd yet. Execution does NOT confirm wolf identity —
  it's just a village vote.

You produce JSON only. No markdown, no commentary.
"""


def _format_retrieved_posts(per_player: Dict[str, List[str]]) -> str:
    blocks = []
    for player, posts in per_player.items():
        if not posts:
            blocks.append(f"### {player}\n(no posts retrieved)\n")
            continue
        body = "\n".join(f"  - {p}" for p in posts)
        blocks.append(f"### {player}\n{body}\n")
    return "\n".join(blocks)


def _summarize_evidence(evidence: Dict) -> str:
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
        lines.append(f"night_deaths (NOT werewolves): {nd}")
    if ex:
        lines.append(f"executions (just village votes, not confirmation): {ex}")
    return "\n".join(f"- {l}" for l in lines)


_VARIANT_FOCUS = {
    "general": (
        "GENERAL reasoning protocol — weigh all signals evenly:\n"
        "  - hard evidence (night_death = NOT wolf, trusted Medium results)\n"
        "  - claim conflicts\n"
        "  - voting & defensive patterns\n"
        "  - language consistency"
    ),
    "voting": (
        "VOTING / ACCUSATION focus — read retrieved posts carefully and look for:\n"
        "  - who votes whom (especially players who AVOID voting confirmed wolves)\n"
        "  - who defends suspected wolves vs who challenges them\n"
        "  - who attacks claimed Seers/Mediums (wolves often try to discredit special roles)\n"
        "  - bandwagon vs independent vote patterns\n"
        "Wolves tend to: defend each other subtly, attack the real Seer/Medium,\n"
        "vote for villagers under pressure, and follow each other's lead."
    ),
    "madman": (
        "MADMAN-vs-WEREWOLF focus — be especially careful about role assignment:\n"
        "  - the Madman is HUMAN but acts wolf-aligned; wolf_score for Madman MUST be LOW (0.05–0.25)\n"
        "  - Madman often: fakes Seer/Medium CO, gives misleading divinations,\n"
        "    pushes lynch votes against village-aligned players, sows confusion\n"
        "  - True Werewolves are more coordinated, less chaotic — they share night info\n"
        "  - When a fake-Seer's white list contains a Medium-confirmed wolf → that\n"
        "    player is fake (Madman OR Werewolf). Look at chat tone:\n"
        "      * chaotic / fact-shifting / over-the-top → lean Madman (low wolf_score)\n"
        "      * methodical / coordinating with other suspects → lean Werewolf\n"
        "  - List ALL suspicious players who look more Madman than wolf in madman_like_players."
    ),
}


def build_user_prompt_rag(
    record: Dict,
    retrieved_posts: Dict[str, List[str]],
    variant: str = "general",
) -> str:
    players = record["players_csv"]
    role_counts = record.get("role_counts_used", {})
    evidence = record["evidence"]

    rc_str = ", ".join(f"{r}:{n}" for r, n in role_counts.items())
    n_wolves = role_counts.get("Werewolf", 0)
    focus_block = _VARIANT_FOCUS.get(variant, _VARIANT_FOCUS["general"])

    schema = {
        "wolf_ranking": players[:3] + ["..."],
        "madman_like_players": [],
        "players": {
            players[0] if players else "X": {
                "wolf_score": 0.1,
                "reason": "short one-line reason",
            }
        }
    }

    return f"""GAME {record['game_idx']} (split={record['split']}, variant={variant})

Players to predict ({len(players)}): {players}

Role configuration: {rc_str}  → exactly {n_wolves} TRUE werewolves.

Structured evidence:
{_summarize_evidence(evidence)}

Retrieved posts per player (own posts + posts mentioning them + reply chain
+ semantic search):

{_format_retrieved_posts(retrieved_posts)}

=== Reasoning protocol ===

{focus_block}

Then in order:
Step 1. Apply hard constraints:
  - night_death victims → wolf_score = 0.0
  - trusted Medium's black list → near 1.0 wolf_score
  - trusted Medium's white list → near 0.0 wolf_score

Step 2. Identify Madman candidates and list them in madman_like_players.
   Madman wolf_score stays LOW (0.05–0.25) — they are HUMAN.

Step 3. Produce wolf_ranking — an ordered list of ALL {len(players)} players
from MOST LIKELY true werewolf to LEAST. Top {n_wolves} entries should be
your wolf candidates.

Step 4. Assign per-player wolf_score consistent with the ranking (higher
rank → higher score, with separation between ranks, especially between the
{n_wolves}-th and ({n_wolves}+1)-th players ≥ 0.15).

=== Output schema (JSON only) ===
{json.dumps(schema, indent=2)}

Rules:
- wolf_ranking length = {len(players)}, all players exactly once.
- Each player has a wolf_score in [0, 1].
- madman_like_players is a subset of `players` that look suspicious but are
  more likely Madman than Werewolf.

Output JSON only, no markdown."""


def build_messages_rag(
    record: Dict,
    retrieved_posts: Dict[str, List[str]],
    variant: str = "general",
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": build_user_prompt_rag(record, retrieved_posts, variant=variant)},
    ]
