"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { PROFILE_FIELDS } from "@/lib/survey-config";

type Props = {
  participantId: string;
};

export function ProfileForm({ participantId }: Props) {
  const router = useRouter();
  const [values, setValues] = useState<Record<string, string>>({});
  const [headphones, setHeadphones] = useState<boolean | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // All fields optional, but require at least one answer to avoid empty submits.
  const answeredAny =
    Object.values(values).some(Boolean) || headphones != null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/survey/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          participant_id: participantId,
          ...values,
          used_headphones: headphones,
        }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b.error ?? `Server returned ${res.status}`);
      }
      const data = (await res.json()) as { next?: string };
      router.push(data.next ?? `/clip-survey/rate?id=${participantId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="mt-10 space-y-8" noValidate>
      {PROFILE_FIELDS.map((field) => (
        <fieldset key={field.key} className="space-y-3">
          <legend className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            {field.label}
          </legend>
          <div className="flex flex-wrap gap-2">
            {field.options.map((opt) => {
              const active = values[field.key] === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() =>
                    setValues((p) => ({
                      ...p,
                      [field.key]: active ? "" : opt.value,
                    }))
                  }
                  aria-pressed={active}
                  className={`h-10 rounded-sm border px-4 text-sm transition-colors ${
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

      {/* Headphones — boolean */}
      <fieldset className="space-y-3">
        <legend className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          作答時是否使用耳機
        </legend>
        <div className="flex flex-wrap gap-2">
          {[
            { v: true, label: "是" },
            { v: false, label: "否" },
          ].map((opt) => {
            const active = headphones === opt.v;
            return (
              <button
                key={opt.label}
                type="button"
                onClick={() => setHeadphones(active ? null : opt.v)}
                aria-pressed={active}
                className={`h-10 rounded-sm border px-4 text-sm transition-colors ${
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

      <div className="flex flex-col gap-3">
        <button
          type="submit"
          disabled={submitting}
          className="inline-flex h-12 w-full max-w-xs items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-[1rem] text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "儲存中…" : "開始評分"}
        </button>
        {!answeredAny && (
          <p className="text-sm text-[color:var(--text-muted)]">
            這些欄位皆為選填，但建議至少填一項以利分析。
          </p>
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
