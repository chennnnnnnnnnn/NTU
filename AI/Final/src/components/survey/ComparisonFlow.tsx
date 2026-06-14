"use client";

import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { AudioPlayer } from "@/components/AudioPlayer";
import { COMPARISON_QUESTIONS, SCALE_MIN, SCALE_MAX, SHOW_SENTENCE } from "@/lib/survey-config";

export type FlowItem = {
  item_id: string;
  order_index: number;
  item_type: "attention" | "comparison";
  answered: boolean;
  question_type?: string;
  sentence?: string | null;
  clip_a_url?: string;
  clip_b_url?: string;
  prompt?: string;
};

type Props = {
  participantId: string;
  items: FlowItem[];
};

const QMAP = Object.fromEntries(COMPARISON_QUESTIONS.map((q) => [q.type, q]));
const SCALE = Array.from({ length: SCALE_MAX - SCALE_MIN + 1 }, (_, i) => SCALE_MIN + i);

export function ComparisonFlow({ participantId, items }: Props) {
  const router = useRouter();
  const total = items.length;

  const firstUnanswered = items.findIndex((it) => !it.answered);
  const [currentIdx, setCurrentIdx] = useState(firstUnanswered === -1 ? 0 : firstUnanswered);
  const [answered, setAnswered] = useState<Record<string, boolean>>(
    Object.fromEntries(items.map((it) => [it.item_id, it.answered])),
  );
  const [choice, setChoice] = useState<Record<string, number>>({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const item = items[currentIdx];
  const isComparison = item.item_type === "comparison";
  const q = isComparison && item.question_type ? QMAP[item.question_type] : null;
  const picked = choice[item.item_id];
  const answeredCount = useMemo(
    () => Object.values(answered).filter(Boolean).length,
    [answered],
  );

  const handleSubmit = useCallback(async () => {
    if (typeof picked !== "number" || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/survey/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ participant_id: participantId, item_id: item.item_id, rating: picked }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.error ?? `Server returned ${res.status}`);
      }
      const data = (await res.json()) as { completed: boolean; next: string | null };
      setAnswered((p) => ({ ...p, [item.item_id]: true }));
      if (data.completed && data.next) {
        router.push(data.next);
        return;
      }
      const nextIdx = items.findIndex((it, i) => i > currentIdx && !answered[it.item_id]);
      setCurrentIdx(nextIdx === -1 ? Math.min(total - 1, currentIdx + 1) : nextIdx);
      setSubmitting(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }, [picked, submitting, participantId, item, items, currentIdx, answered, total, router]);

  return (
    <div className="mt-10 space-y-10">
      <div className="flex items-baseline justify-between">
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          已完成 {answeredCount} / {total}
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">第 {currentIdx + 1} / {total} 題</p>
      </div>

      {isComparison && q ? (
        <div className="space-y-6">
          {/* A / B players */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-3 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-card)] p-5">
              <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">語音 A</p>
              <AudioPlayer key={`${item.item_id}-a`} src={item.clip_a_url ?? null} />
            </div>
            <div className="space-y-3 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-card)] p-5">
              <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">語音 B</p>
              <AudioPlayer key={`${item.item_id}-b`} src={item.clip_b_url ?? null} />
            </div>
          </div>

          {SHOW_SENTENCE && item.sentence && (
            <p className="font-serif text-[1.0625rem] text-[color:var(--text-ink)]">
              &ldquo;{item.sentence}&rdquo;
            </p>
          )}

          {/* Question + bipolar 1–5 scale */}
          <fieldset className="space-y-3">
            <legend className="text-[1.0625rem] text-[color:var(--text-ink)]">{q.prompt}</legend>
            <div className="flex items-stretch gap-2">
              {SCALE.map((v) => {
                const active = picked === v;
                return (
                  <button
                    key={v}
                    type="button"
                    onClick={() => setChoice((p) => ({ ...p, [item.item_id]: v }))}
                    aria-pressed={active}
                    className={`mono inline-flex h-12 flex-1 items-center justify-center rounded-sm border text-[1rem] transition-colors ${
                      active
                        ? "border-[color:var(--accent)] bg-[color:var(--accent)] text-white"
                        : "border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] text-[color:var(--text-ink)] hover:border-[color:var(--border-strong)]"
                    }`}
                  >
                    {v}
                  </button>
                );
              })}
            </div>
            <div className="flex justify-between">
              <span className="text-xs text-[color:var(--text-muted)]">A {q.endLabel}</span>
              <span className="text-xs text-[color:var(--text-muted)]">B {q.endLabel}</span>
            </div>
          </fieldset>
        </div>
      ) : (
        /* Attention check */
        <fieldset className="space-y-3 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-card)] p-6">
          <legend className="text-[1.0625rem] text-[color:var(--text-ink)]">{item.prompt}</legend>
          <div className="flex items-stretch gap-2">
            {SCALE.map((v) => {
              const active = picked === v;
              return (
                <button
                  key={v}
                  type="button"
                  onClick={() => setChoice((p) => ({ ...p, [item.item_id]: v }))}
                  aria-pressed={active}
                  className={`mono inline-flex h-12 flex-1 items-center justify-center rounded-sm border text-[1rem] transition-colors ${
                    active
                      ? "border-[color:var(--accent)] bg-[color:var(--accent)] text-white"
                      : "border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] text-[color:var(--text-ink)] hover:border-[color:var(--border-strong)]"
                  }`}
                >
                  {v}
                </button>
              );
            })}
          </div>
        </fieldset>
      )}

      {/* Submit */}
      <div className="flex flex-col gap-3 border-t border-[color:var(--border-warm)] pt-6">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={typeof picked !== "number" || submitting}
          className="inline-flex h-12 w-full max-w-xs items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-[1rem] text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "儲存中…" : currentIdx === total - 1 ? "完成並送出" : "送出並繼續 →"}
        </button>
        {typeof picked !== "number" && (
          <p className="text-sm text-[color:var(--text-muted)]">請先選擇 1–5 再繼續。</p>
        )}
        {error && (
          <p role="alert" className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]">
            {error}
          </p>
        )}
      </div>

      {/* Progress dots */}
      <ol className="flex flex-wrap gap-1.5 border-t border-[color:var(--border-warm)] pt-5" aria-label="作答進度">
        {items.map((it, i) => {
          const isAns = answered[it.item_id];
          const cls = isAns
            ? "bg-[color:var(--success)] border-[color:var(--success)] text-white"
            : i === currentIdx
            ? "bg-[color:var(--accent)] border-[color:var(--accent)] text-white"
            : "bg-[color:var(--bg-elevated)] border-[color:var(--border-warm)] text-[color:var(--text-muted)]";
          return (
            <li key={it.item_id}>
              <button
                type="button"
                onClick={() => setCurrentIdx(i)}
                aria-label={`前往第 ${i + 1} 題`}
                aria-current={i === currentIdx ? "step" : undefined}
                className={`mono inline-flex h-7 w-7 items-center justify-center rounded-sm border text-[10px] ${cls}`}
              >
                {i + 1}
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
