"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const LIKERT_ITEMS = [
  {
    key: "self_efficacy",
    prompt:
      "After this session, I feel more confident reading English aloud.",
  },
  {
    key: "perceived_improvement",
    prompt: "I think my English pronunciation improved during this session.",
  },
  {
    key: "naturalness_c1",
    prompt: "The native model voices (C1) sounded natural to me.",
  },
  {
    key: "naturalness_c2",
    prompt:
      "The self-voice with native accent (C2) sounded natural to me.",
  },
  {
    key: "naturalness_c3",
    prompt:
      "The self-voice with L1 accent (C3) sounded natural to me.",
  },
  {
    key: "engagement",
    prompt: "I was engaged throughout this session.",
  },
] as const;

const LIKERT_LABELS = [
  "Strongly disagree",
  "Disagree",
  "Neutral",
  "Agree",
  "Strongly agree",
];

const RANK_LABELS: Record<1 | 2 | 3, string> = {
  1: "C1 · Native model voice",
  2: "C2 · Self voice, native accent",
  3: "C3 · Self voice, L1 accent",
};

type RankItem = 1 | 2 | 3;

type Props = {
  participantId: string;
};

export function SurveyForm({ participantId }: Props) {
  const router = useRouter();
  const [likert, setLikert] = useState<Record<string, number>>({});
  const [ranking, setRanking] = useState<RankItem[]>([]);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const allLikertAnswered = LIKERT_ITEMS.every((it) => likert[it.key] != null);
  const rankComplete = ranking.length === 3;
  const canSubmit = allLikertAnswered && rankComplete && !submitting;

  function toggleRank(c: RankItem) {
    setRanking((prev) => {
      const idx = prev.indexOf(c);
      if (idx !== -1) {
        // Remove from ranking (un-rank)
        return prev.filter((x) => x !== c);
      }
      if (prev.length >= 3) return prev;
      return [...prev, c];
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/submit-survey", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          participant_id: participantId,
          likert,
          ranking,
          comment: comment.trim() || null,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Server returned ${res.status}`);
      }
      router.push(`/done/complete?id=${participantId}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-10" noValidate>
      {/* Likert */}
      <section className="space-y-6">
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Likert · seven statements
        </p>
        {LIKERT_ITEMS.map((it) => (
          <fieldset key={it.key} className="space-y-3 border-b border-[color:var(--border-warm)] pb-5">
            <legend
              style={{ color: "var(--text-ink)" }}
              className="font-serif text-base"
            >
              {it.prompt}
            </legend>
            <div className="flex flex-wrap gap-2">
              {LIKERT_LABELS.map((label, i) => {
                const v = i + 1;
                const selected = likert[it.key] === v;
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setLikert((p) => ({ ...p, [it.key]: v }))}
                    aria-pressed={selected}
                    style={
                      selected
                        ? { backgroundColor: "var(--accent)", borderColor: "var(--accent)", color: "#ffffff" }
                        : { backgroundColor: "var(--bg-elevated)", borderColor: "var(--border-warm)", color: "var(--text-muted)" }
                    }
                    className="inline-flex h-10 items-center justify-center rounded-sm border px-3 mono text-xs hover:opacity-80"
                  >
                    {v} · {label}
                  </button>
                );
              })}
            </div>
          </fieldset>
        ))}
      </section>

      {/* Ranking */}
      <section className="space-y-4">
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Ranking · most helpful first
        </p>
        <p className="text-sm text-[color:var(--text-muted)]">
          Tap each condition in the order you found most helpful for learning.
          Tap again to un-rank.
        </p>
        <div className="space-y-2">
          {([1, 2, 3] as RankItem[]).map((c) => {
            const rank = ranking.indexOf(c);
            return (
              <button
                key={c}
                type="button"
                onClick={() => toggleRank(c)}
                style={
                  rank !== -1
                    ? { backgroundColor: "var(--accent-soft)", borderColor: "var(--accent)" }
                    : { backgroundColor: "var(--bg-elevated)", borderColor: "var(--border-warm)" }
                }
                className="flex w-full items-center gap-4 rounded-sm border p-3 text-left transition-colors hover:opacity-90"
              >
                <span
                  style={{ borderColor: "var(--border-strong)", backgroundColor: "var(--bg-base)", color: "var(--text-ink)" }}
                  className="mono inline-flex h-8 w-8 items-center justify-center rounded-sm border text-sm"
                >
                  {rank !== -1 ? rank + 1 : "—"}
                </span>
                <span style={{ color: "var(--text-ink)" }}>{RANK_LABELS[c]}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* Optional free text */}
      <section className="space-y-3">
        <label htmlFor="comment" className="mono block text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Anything else (optional)
        </label>
        <textarea
          id="comment"
          rows={4}
          value={comment}
          onChange={(e) => setComment(e.target.value.slice(0, 1000))}
          className="block w-full rounded-sm border border-[color:var(--border-strong)] bg-[color:var(--bg-elevated)] px-3 py-2 text-sm text-[color:var(--text-ink)] placeholder:text-[color:var(--text-faint)]"
          placeholder="What was your experience like?"
        />
        <p className="mono text-xs text-[color:var(--text-faint)]">
          {comment.length} / 1000
        </p>
      </section>

      {error && (
        <p role="alert" className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="inline-flex h-12 w-full max-w-sm items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        {submitting ? "Submitting…" : "Submit survey"}
      </button>

      {!allLikertAnswered && (
        <p className="text-sm text-[color:var(--text-muted)]">
          Please answer all six Likert items.
        </p>
      )}
      {!rankComplete && (
        <p className="text-sm text-[color:var(--text-muted)]">
          Please rank all three conditions.
        </p>
      )}
    </form>
  );
}
