"""Parse role-count announcement from each game log.

The log contains a line like:
    "It seems there are nine villagers, three werewolves, one seer,
     one medium, and one madman, along with one hunter."

Smaller games may omit Hunter / Madman:
    "...six villagers, two werewolves, one seer, and one medium among us."

Usage as module:
    from parsers.role_counts import parse_role_counts, parse_all
"""
import argparse
import json
import re
from pathlib import Path

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16,
}
NUM = r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|\d+)"

ROLE_PATTERNS = {
    "Villager": rf"\b{NUM}\s+villagers?\b",
    "Werewolf": rf"\b{NUM}\s+werewolves?\b",
    "Seer":     rf"\b{NUM}\s+seer\b",
    "Medium":   rf"\b{NUM}\s+medium\b",
    "Madman":   rf"\b{NUM}\s+madman\b",
    "Hunter":   rf"\b{NUM}\s+hunter\b",
}

# Header sentence must look like an official announcement.
HEADER_HINT = re.compile(
    r"it seems there are .*villagers?.*werewolves?",
    re.IGNORECASE | re.DOTALL,
)


def _word_to_int(word: str) -> int:
    word = word.lower()
    if word.isdigit():
        return int(word)
    return NUM_WORDS.get(word, 0)


def parse_role_counts(text: str) -> dict:
    """Find the official role-count line and parse it.

    Returns dict with keys role->count for the roles present.
    Returns {} if no header sentence is found (no-header case).
    """
    # Find the header line. Search lines that match the hint.
    target_line = None
    for line in text.splitlines():
        if "it seems there are" in line.lower() and re.search(
            r"villagers?", line, re.IGNORECASE
        ) and re.search(r"werewolves?|werewolf", line, re.IGNORECASE):
            target_line = line
            break

    if not target_line:
        return {}

    lower = target_line.lower()
    counts = {}
    for role, pattern in ROLE_PATTERNS.items():
        m = re.search(pattern, lower)
        if m:
            counts[role] = _word_to_int(m.group(1))
    return counts


def classify_failure(text: str, counts: dict) -> str:
    """Return 'ok' / 'no_header' / 'parser_failed'."""
    if counts and "Villager" in counts and "Werewolf" in counts:
        return "ok"
    has_hint = bool(HEADER_HINT.search(text))
    if not has_hint:
        return "no_header"
    return "parser_failed"


def parse_all(data_dir: Path, splits=("public", "private")) -> dict:
    """Parse role counts for every game under data_dir/<split>/*.txt."""
    out = {}
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for log_path in sorted(split_dir.glob("*.txt")):
            text = log_path.read_text(encoding="utf-8", errors="replace")
            counts = parse_role_counts(text)
            status = classify_failure(text, counts)
            out[f"{split}/{log_path.stem}"] = {
                "counts": counts,
                "status": status,
            }
    return out


def coverage_report(results: dict) -> str:
    total = len(results)
    ok = [k for k, v in results.items() if v["status"] == "ok"]
    no_header = [k for k, v in results.items() if v["status"] == "no_header"]
    parser_failed = [k for k, v in results.items() if v["status"] == "parser_failed"]

    lines = [
        f"Total games: {total}",
        f"Parsed successfully:        {len(ok)}",
        f"Fallback - no header found: {len(no_header)}",
        f"Fallback - parser failed:   {len(parser_failed)}",
    ]
    if no_header:
        lines.append("")
        lines.append("No-header games:")
        for k in no_header:
            lines.append(f"  - {k}")
    if parser_failed:
        lines.append("")
        lines.append("Parser-failed games:")
        for k in parser_failed:
            lines.append(f"  - {k}: counts={results[k]['counts']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/role_counts/role_counts.json"),
    )
    args = ap.parse_args()

    results = parse_all(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(coverage_report(results))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
