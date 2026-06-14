"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { BATCH_SELECTORS } from "@/lib/survey-config";

export function StartForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [agreed, setAgreed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nameOk = name.trim().length > 0;
  const allSelected = BATCH_SELECTORS.every((s) => selections[s.key]);
  const canSubmit = agreed && nameOk && allSelected && !submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/survey/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          team_member_name: name.trim(),
          person_num: selections.person_num,
        }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.error ?? `Server returned ${res.status}`);
      }
      const data = (await res.json()) as { participant_id: string; next?: string };
      router.push(data.next ?? `/clip-survey/profile?id=${data.participant_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8" noValidate>
      {/* Visitor name (required) */}
      <div className="space-y-2">
        <label
          htmlFor="visitor-name"
          className="mono block text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]"
        >
          組員姓名（必填）
        </label>
        <input
          id="visitor-name"
          type="text"
          autoComplete="off"
          value={name}
          onChange={(e) => setName(e.target.value.slice(0, 60))}
          className="block w-full max-w-xs rounded-sm border border-[color:var(--border-strong)] bg-[color:var(--bg-elevated)] px-3 py-2 text-[1.0625rem] text-[color:var(--text-ink)] placeholder:text-[color:var(--text-faint)]"
          placeholder="請輸入您的姓名"
        />
      </div>

      {/* Batch selectors (第幾人) */}
      {BATCH_SELECTORS.map((sel) => (
        <fieldset key={sel.key} className="space-y-3">
          <legend className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            {sel.label}
          </legend>
          <div className="flex flex-wrap gap-2">
            {sel.options.map((opt) => {
              const active = selections[sel.key] === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() =>
                    setSelections((p) => ({ ...p, [sel.key]: active ? "" : opt.value }))
                  }
                  aria-pressed={active}
                  className={`mono h-10 rounded-sm border px-4 text-sm transition-colors ${
                    active
                      ? "border-[color:var(--accent)] bg-[color:var(--accent)] text-white"
                      : "border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] text-[color:var(--text-ink)] hover:border-[color:var(--border-strong)]"
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </fieldset>
      ))}

      {/* Consent */}
      <label
        htmlFor="survey-consent"
        className="flex cursor-pointer items-start gap-3 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] p-3 text-sm leading-relaxed transition-colors hover:border-[color:var(--border-strong)]"
      >
        <input
          id="survey-consent"
          type="checkbox"
          checked={agreed}
          onChange={(e) => setAgreed(e.target.checked)}
          className="mt-[3px] h-4 w-4 shrink-0 accent-[color:var(--accent)]"
        />
        <span className="text-[color:var(--text-ink)]">
          我了解這份語音評分問卷的填答僅用於學術研究，且我可隨時停止。
        </span>
      </label>

      <div className="flex flex-col gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="inline-flex h-12 w-full max-w-xs items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-[1rem] text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "進入中…" : "開始"}
        </button>
        {!nameOk && <p className="text-sm text-[color:var(--text-muted)]">請填寫組員姓名。</p>}
        {nameOk && !allSelected && (
          <p className="text-sm text-[color:var(--text-muted)]">請選擇「第幾人」。</p>
        )}
        {nameOk && allSelected && !agreed && (
          <p className="text-sm text-[color:var(--text-muted)]">請勾選同意項目以繼續。</p>
        )}
        {error && (
          <p
            role="alert"
            className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
          >
            {error}
          </p>
        )}
      </div>
    </form>
  );
}
