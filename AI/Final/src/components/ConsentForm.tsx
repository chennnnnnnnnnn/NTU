"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const CONSENT_ITEMS: { id: string; text: string }[] = [
  {
    id: "voluntary",
    text: "I understand that participation is voluntary and I may stop at any time without penalty.",
  },
  {
    id: "recording",
    text: "I consent to having my speech recorded (Mandarin and English) and stored on a private research server.",
  },
  {
    id: "ai-cloning",
    text: "I consent to my voice being processed by third-party voice-cloning services (Fish Audio, ElevenLabs) for the purpose of this study only.",
  },
  {
    id: "data-use",
    text: "I understand my recordings will be analysed for academic research, may appear in anonymised form in publications, and will not be sold or used for any commercial purpose.",
  },
];

const CONSENT_VERSION = "v1.0";

type Props = {
  initialCode: string;
};

export function ConsentForm({ initialCode }: Props) {
  const router = useRouter();
  const [code, setCode] = useState(initialCode);
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const allChecked = CONSENT_ITEMS.every((item) => checked[item.id]);
  const codeValid = /^[A-Z0-9]{6,12}$/.test(code);
  const canSubmit = allChecked && codeValid && !submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code,
          consent_version: CONSENT_VERSION,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Server returned ${res.status}`);
      }
      const data = (await res.json()) as {
        participant_id: string;
        next?: string;
        status?: string;
      };
      // Resume to wherever the participant left off (server decides).
      router.push(data.next ?? `/calibration?id=${data.participant_id}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8" noValidate>
      {/* Invitation code */}
      <div className="space-y-2">
        <label
          htmlFor="invitation-code"
          className="mono block text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]"
        >
          Invitation code
        </label>
        <input
          id="invitation-code"
          name="code"
          type="text"
          inputMode="text"
          autoCapitalize="characters"
          autoComplete="off"
          spellCheck={false}
          value={code}
          onChange={(e) => setCode(e.target.value.toUpperCase().trim().slice(0, 12))}
          className="block w-full max-w-xs rounded-sm border border-[color:var(--border-strong)] bg-[color:var(--bg-elevated)] px-3 py-2 mono text-[1.125rem] tracking-[0.12em] text-[color:var(--text-ink)] placeholder:text-[color:var(--text-faint)]"
          placeholder="ABC123XY"
          aria-describedby="code-hint"
        />
        <p id="code-hint" className="text-xs text-[color:var(--text-muted)]">
          6–12 uppercase letters or digits. Provided in your invitation email.
        </p>
      </div>

      {/* Consent checklist */}
      <fieldset className="space-y-3">
        <legend className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Consent
        </legend>
        {CONSENT_ITEMS.map((item) => (
          <label
            key={item.id}
            htmlFor={`consent-${item.id}`}
            className="flex cursor-pointer items-start gap-3 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] p-3 text-sm leading-relaxed transition-colors hover:border-[color:var(--border-strong)]"
          >
            <input
              id={`consent-${item.id}`}
              type="checkbox"
              checked={Boolean(checked[item.id])}
              onChange={(e) =>
                setChecked((prev) => ({ ...prev, [item.id]: e.target.checked }))
              }
              className="mt-[3px] h-4 w-4 shrink-0 accent-[color:var(--accent)]"
            />
            <span className="text-[color:var(--text-ink)]">{item.text}</span>
          </label>
        ))}
      </fieldset>

      {/* Submit */}
      <div className="flex flex-col gap-3">
        <button
          type="submit"
          disabled={!canSubmit}
          className="inline-flex h-12 w-full max-w-xs items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "Enrolling…" : "Begin calibration"}
        </button>

        {!codeValid && code.length > 0 && (
          <p className="text-sm text-[color:var(--danger)]" role="alert">
            Code must be 6–12 uppercase letters or digits.
          </p>
        )}
        {!allChecked && (
          <p className="text-sm text-[color:var(--text-muted)]">
            Please tick all consent items to continue.
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
