"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Progress = { total: number; done: number; failed: number; errors: string[] };

type Props = {
  participantId: string;
  recordingsSaved: number;
  recordingsExpected: number;
};

const POLL_INTERVAL_MS = 4000;
const MAX_WAIT_MS = 8 * 60 * 1000;

export function PreTestDoneFlow({
  participantId,
  recordingsSaved,
  recordingsExpected,
}: Props) {
  const preTestComplete =
    recordingsExpected > 0 && recordingsSaved === recordingsExpected;

  const [progress, setProgress] = useState<Progress | null>(null);
  const [phase, setPhase] = useState<
    "idle" | "starting" | "running" | "ready" | "error"
  >("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const startedAtRef = useRef<number | null>(null);
  const triggeredRef = useRef(false);

  // Kick off generation once the pre-test is complete and we haven't kicked yet.
  const triggerGeneration = useCallback(async () => {
    if (triggeredRef.current) return;
    triggeredRef.current = true;
    startedAtRef.current = Date.now();
    setPhase("starting");
    setErrorMsg(null);
    try {
      const res = await fetch("/api/generate-stimuli", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ participant_id: participantId }),
      });
      const body = (await res.json().catch(() => ({}))) as Progress & { error?: string };
      if (!res.ok) {
        // 409 (calibration not complete) or 503 (missing keys) — show, don't poll
        setErrorMsg(body.error ?? `Server returned ${res.status}`);
        setPhase("error");
        return;
      }
      setProgress(body);
      if (body.done === body.total) {
        setPhase("ready");
      } else if (body.failed > 0) {
        setErrorMsg(
          `Some clips failed (${body.failed}/${body.total}). First error: ${body.errors[0] ?? "unknown"}`,
        );
        setPhase("error");
      } else {
        setPhase("running");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setPhase("error");
    }
  }, [participantId]);

  useEffect(() => {
    if (!preTestComplete) return;
    void triggerGeneration();
  }, [preTestComplete, triggerGeneration]);

  // Poll every POLL_INTERVAL_MS while running.
  useEffect(() => {
    if (phase !== "running" && phase !== "starting") return;
    let cancelled = false;
    const id = window.setInterval(async () => {
      if (cancelled) return;
      const startedAt = startedAtRef.current ?? Date.now();
      if (Date.now() - startedAt > MAX_WAIT_MS) {
        window.clearInterval(id);
        setErrorMsg(
          "Generation is taking longer than expected. Please notify the researcher.",
        );
        setPhase("error");
        return;
      }
      try {
        const res = await fetch(
          `/api/generate-stimuli?id=${encodeURIComponent(participantId)}`,
        );
        const body = (await res.json()) as Progress;
        if (cancelled) return;
        setProgress(body);
        if (body.done === body.total && body.total > 0) {
          setPhase("ready");
          window.clearInterval(id);
        }
      } catch {
        // Network blip — keep polling; the POST response is the source of truth.
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [phase, participantId]);

  if (!preTestComplete) {
    return (
      <p
        role="alert"
        className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
      >
        Some sentences are missing recordings ({recordingsSaved}/{recordingsExpected} saved).
        Please return to the{" "}
        <a
          href={`/pre-test?id=${participantId}`}
          className="text-[color:var(--accent)] underline-offset-4 hover:underline"
        >
          pre-test
        </a>{" "}
        and complete them before continuing.
      </p>
    );
  }

  // pre-test complete — show generation progress / outcome
  const total = progress?.total ?? 0;
  const done = progress?.done ?? 0;

  return (
    <div className="space-y-4 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] p-5">
      <p className="font-serif text-lg">Preparing your model voices</p>
      <p className="text-sm text-[color:var(--text-muted)]">
        Two services are now generating your training audio: ElevenLabs renders
        your voice with a native accent, and Fish Audio renders it with your L1
        Mandarin accent leak. This usually finishes within one to two minutes.
      </p>

      <div className="space-y-2">
        <div className="flex items-baseline justify-between mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          <span>Generation progress</span>
          <span>
            {done} / {total > 0 ? total : "…"}
          </span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-card)]">
          <div
            style={{
              width: total > 0 ? `${(done / total) * 100}%` : "0%",
              backgroundColor: "var(--accent)",
              transition: "width 200ms linear",
            }}
            className="h-full"
            aria-hidden
          />
        </div>
      </div>

      {phase === "ready" && (
        <div className="space-y-3 pt-1">
          <p className="text-sm text-[color:var(--success)]">
            All training clips are ready.
          </p>
          <a
            href={`/train?id=${participantId}`}
            style={{
              backgroundColor: "var(--accent)",
              borderColor: "var(--accent)",
              color: "#ffffff",
            }}
            className="inline-flex h-12 items-center justify-center rounded-sm border px-6 text-base font-medium transition-opacity hover:opacity-90"
          >
            Continue to training →
          </a>
        </div>
      )}

      {phase === "error" && errorMsg && (
        <div
          role="alert"
          className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
        >
          <p className="font-medium">Something went wrong while preparing audio</p>
          <p className="mono text-xs">{errorMsg}</p>
          <button
            type="button"
            onClick={() => {
              triggeredRef.current = false;
              void triggerGeneration();
            }}
            className="mt-2 text-sm text-[color:var(--accent)] underline-offset-4 hover:underline"
          >
            Try again
          </button>
        </div>
      )}

      {(phase === "starting" || phase === "running") && (
        <p className="text-xs text-[color:var(--text-muted)]">
          You can keep this tab open and look away — we will route you forward
          automatically when the clips are ready.
        </p>
      )}

      <p className="mono text-xs text-[color:var(--text-faint)]">
        participant_id · {participantId}
      </p>
    </div>
  );
}
