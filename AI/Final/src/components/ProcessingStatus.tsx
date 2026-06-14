"use client";

import { useEffect, useRef, useState } from "react";

type Status = {
  fish_ready: boolean;
  elevenlabs_ready: boolean;
  participant_status: string;
  error?: string;
};

const POLL_INTERVAL_MS = 4000;
const MAX_POLL_MS = 5 * 60 * 1000; // 5 minutes hard ceiling

export function ProcessingStatus({ participantId }: { participantId: string }) {
  const [status, setStatus] = useState<Status>({
    fish_ready: false,
    elevenlabs_ready: false,
    participant_status: "enrolled",
  });
  const [pollError, setPollError] = useState<string | null>(null);
  const startedAtRef = useRef<number>(Date.now());

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      try {
        const res = await fetch(
          `/api/finalize-voice?id=${encodeURIComponent(participantId)}`,
          { method: "GET" },
        );
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error ?? `Server returned ${res.status}`);
        }
        const data = (await res.json()) as Status;
        if (!cancelled) setStatus(data);
      } catch (err) {
        if (!cancelled) {
          setPollError(err instanceof Error ? err.message : String(err));
        }
      }
    };

    void tick();
    const id = window.setInterval(() => {
      if (Date.now() - startedAtRef.current > MAX_POLL_MS) {
        window.clearInterval(id);
        setPollError(
          "Voice processing is taking longer than expected. Please contact the researcher.",
        );
        return;
      }
      void tick();
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [participantId]);

  const bothReady = status.fish_ready && status.elevenlabs_ready;

  return (
    <div className="mt-10 space-y-8">
      <ul className="space-y-3" aria-live="polite">
        <ProgressRow
          label="Fish Audio · cross-lingual model"
          ready={status.fish_ready}
        />
        <ProgressRow
          label="ElevenLabs · voice changer model"
          ready={status.elevenlabs_ready}
        />
      </ul>

      {pollError && (
        <p
          role="alert"
          className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
        >
          {pollError}
        </p>
      )}

      {bothReady && (
        <div className="space-y-4 rounded-sm border border-[color:var(--success)] bg-[color:var(--bg-elevated)] p-5">
          <p className="font-serif text-lg">Calibration complete.</p>
          <p className="text-sm text-[color:var(--text-muted)]">
            Your voice models are ready. The next stage is pre-test reading:
            you will read 18 short English sentences aloud at your natural
            pace, without hearing any model first.
          </p>
          <a
            href={`/pre-test?id=${participantId}`}
            className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)]"
          >
            Continue to pre-test →
          </a>
          <p className="mono text-xs text-[color:var(--text-faint)]">
            participant_id · {participantId}
          </p>
        </div>
      )}
    </div>
  );
}

function ProgressRow({ label, ready }: { label: string; ready: boolean }) {
  return (
    <li className="flex items-center gap-4 border-b border-[color:var(--border-warm)] py-3 last:border-b-0">
      <span
        aria-hidden
        className={`inline-block h-3 w-3 rounded-full ${
          ready
            ? "bg-[color:var(--success)]"
            : "animate-pulse bg-[color:var(--accent)]"
        }`}
      />
      <span className="text-[color:var(--text-ink)]">{label}</span>
      <span className="ml-auto mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
        {ready ? "ready" : "processing"}
      </span>
    </li>
  );
}
