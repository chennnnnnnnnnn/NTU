"""A3 Reasoning Agent.

Phase 3.1 design:
    - Input: per-game evidence JSON (dumped by dump_evidence.py).
    - LLM: Qwen2.5-7B-Instruct GGUF Q4 via llama-cpp-python.
    - Output: {player: {role_scores, wolf_score, reason}}.
    - Robust JSON parsing with json-repair + retry + fallback (caller falls
      back to v3 rule baseline on hard failure).

Reproducibility (per strategy3.3):
    - temperature = 0
    - top_p = 1
    - fixed seed
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import json_repair
except ImportError:  # pragma: no cover
    json_repair = None  # graceful: caller will see hard fallback

from agents.prompts import build_messages
from agents.prompts_rag import build_messages_rag


ROLES = ["Villager", "Werewolf", "Seer", "Medium", "Hunter", "Madman"]


@dataclass
class ReasonerConfig:
    model_path: str
    n_gpu_layers: int = -1
    n_ctx: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1800
    seed: int = 1234
    verbose: bool = False


class Reasoner:
    """Wraps llama-cpp-python's Llama with a chat-style call."""

    def __init__(self, cfg: ReasonerConfig):
        from llama_cpp import Llama  # lazy import so the module is testable
        self.cfg = cfg
        # chat_format left to auto-detect from GGUF metadata (Qwen2.5 uses chatml).
        self.llm = Llama(
            model_path=cfg.model_path,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.n_ctx,
            seed=cfg.seed,
            verbose=cfg.verbose,
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
        )
        return out["choices"][0]["message"]["content"]


# ---------------- JSON parsing ----------------

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)
_BRACES_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_to_json(text: str) -> str:
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _BRACES_RE.search(text)
    if m:
        return m.group(0).strip()
    return text.strip()


def parse_llm_json(raw: str) -> Optional[Dict]:
    """Strip markdown, try json.loads, fall back to json_repair."""
    candidate = _strip_to_json(raw)
    try:
        return json.loads(candidate)
    except Exception:
        pass
    if json_repair is None:
        return None
    try:
        repaired = json_repair.loads(candidate)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        return None
    return None


# ---------------- Normalize & validate ----------------

def _resolve_player_key(raw_key: str, players: List[str]) -> Optional[str]:
    """Map any LLM-emitted name back to a canonical csv player name."""
    if raw_key in players:
        return raw_key
    key_map_lc = {p.lower(): p for p in players}
    if raw_key.lower() in key_map_lc:
        return key_map_lc[raw_key.lower()]
    # Substring fallback
    for p in players:
        if p.lower() in raw_key.lower() or raw_key.lower() in p.lower():
            return p
    # Token-overlap fallback (handles "Optimist Gerd" vs "Gerd")
    raw_tokens = {t.lower() for t in raw_key.replace("_", " ").split()}
    if raw_tokens:
        best = None
        best_overlap = 0
        for p in players:
            p_tokens = {t.lower() for t in p.split()}
            overlap = len(raw_tokens & p_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = p
        if best_overlap > 0:
            return best
    return None


def normalize_predictions(
    parsed: Dict,
    players: List[str],
) -> Optional[Dict[str, Dict]]:
    """Coerce LLM output into a clean per-player dict.

    Returns None if too many players are missing → caller falls back.
    """
    if not parsed or not isinstance(parsed, dict):
        return None
    players_data = parsed.get("players") if "players" in parsed else parsed
    if not isinstance(players_data, dict):
        return None

    # Optional: wolf_ranking and madman_like_players for RAG prompt.
    wolf_ranking = parsed.get("wolf_ranking") if isinstance(parsed, dict) else None
    madman_like = parsed.get("madman_like_players") if isinstance(parsed, dict) else None
    madman_like_set = set()
    if isinstance(madman_like, list):
        for m in madman_like:
            r = _resolve_player_key(str(m), players)
            if r:
                madman_like_set.add(r)

    ranking_map: Dict[str, int] = {}
    if isinstance(wolf_ranking, list):
        for i, name in enumerate(wolf_ranking):
            r = _resolve_player_key(str(name), players)
            if r and r not in ranking_map:
                ranking_map[r] = i

    out: Dict[str, Dict] = {}
    for raw_key, value in players_data.items():
        if not isinstance(value, dict):
            continue
        canonical = _resolve_player_key(raw_key, players)
        if canonical is None:
            continue

        rs = value.get("role_scores") or {}
        if not isinstance(rs, dict):
            rs = {}
        rs_clean: Dict[str, float] = {r: 0.0 for r in ROLES}
        for k, v in rs.items():
            for r in ROLES:
                if k.lower() == r.lower():
                    try:
                        rs_clean[r] = float(v)
                    except (TypeError, ValueError):
                        rs_clean[r] = 0.0
                    break
        rs_clean = {r: max(0.0, x) for r, x in rs_clean.items()}
        total = sum(rs_clean.values())
        if total > 0:
            rs_clean = {r: x / total for r, x in rs_clean.items()}
        else:
            rs_clean = {r: 1.0 / len(ROLES) for r in ROLES}

        try:
            ws = float(value.get("wolf_score", 0.15))
        except (TypeError, ValueError):
            ws = 0.15
        ws = max(0.0, min(1.0, ws))

        out[canonical] = {
            "role_scores": rs_clean,
            "wolf_score": ws,
            "reason": str(value.get("reason", ""))[:200],
            "madman_like": canonical in madman_like_set,
            "rank": ranking_map.get(canonical),
        }

    if len(out) < max(1, len(players) // 2):
        return None
    return out


def run_one_game(
    reasoner: Reasoner,
    evidence_record: Dict,
    raw_save_path: Optional[Path] = None,
    retrieved_posts: Optional[Dict[str, List[str]]] = None,
    variant: str = "general",
) -> Optional[Dict[str, Dict]]:
    """Run the reasoner on one game; return per-player predictions or None.

    If `retrieved_posts` is supplied → use the RAG prompt (Phase 3.2) with
    the given prompt `variant`. Otherwise use the evidence-only prompt.
    """
    players = evidence_record["players_csv"]
    if retrieved_posts is not None:
        messages = build_messages_rag(evidence_record, retrieved_posts, variant=variant)
    else:
        messages = build_messages(evidence_record)
    raw = reasoner.generate(messages)

    if raw_save_path is not None:
        raw_save_path.parent.mkdir(parents=True, exist_ok=True)
        raw_save_path.write_text(raw, encoding="utf-8")

    parsed = parse_llm_json(raw)
    return normalize_predictions(parsed, players)


if __name__ == "__main__":
    # Smoke test: load model, run on a single evidence JSON, print result.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--ngl", type=int, default=-1)
    args = ap.parse_args()

    cfg = ReasonerConfig(model_path=args.model, n_gpu_layers=args.ngl)
    reasoner = Reasoner(cfg)
    record = json.loads(args.evidence.read_text())
    out = run_one_game(reasoner, record, raw_save_path=Path("/tmp/llm_raw.txt"))
    print(json.dumps(out, indent=2, ensure_ascii=False)[:2000])
