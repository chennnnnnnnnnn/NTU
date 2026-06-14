"use client";

import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { AudioRecorder } from "@/components/AudioRecorder";
import { StimulusPrompt } from "@/components/StimulusPrompt";

const PRE_RECORDING_MIN_SEC = 3;
const PRE_RECORDING_MAX_SEC = 20;
const PRE_RECORDING_WARN_SEC = 15;

type ManifestItem = {
  assignment_id: string;
  stimulus_id: string;
  stimulus_code: string;
  sentence: string;
  syllable_count: number | null;
  target_phonemes: Record<string, number> | null;
  order_index: number;
};

type SegmentStatus = "pending" | "uploading" | "saved" | "error";

type Props = {
  participantId: string;
  manifest: ManifestItem[];
};

export function PreTestFlow({ participantId, manifest }: Props) {
  const router = useRouter();
  const total = manifest.length;
  const [currentIdx, setCurrentIdx] = useState(0);
  const [states, setStates] = useState<Record<string, { status: SegmentStatus; error?: string }>>({});
  const [submitting, setSubmitting] = useState(false);
  const [doneError, setDoneError] = useState<string | null>(null);

  const allSaved = useMemo(
    () => manifest.every((m) => states[m.assignment_id]?.status === "saved"),
    [manifest, states],
  );
  const savedCount = useMemo(
    () => manifest.filter((m) => states[m.assignment_id]?.status === "saved").length,
    [manifest, states],
  );

  const current = manifest[currentIdx];
  const currentStatus = states[current.assignment_id]?.status ?? "pending";
  const currentDone = currentStatus === "saved";

  const handleUpload = useCallback(
    async (assignmentId: string, stimulusCode: string, blob: Blob, durationSec: number, attemptN: number) => {
      setStates((p) => ({ ...p, [assignmentId]: { status: "uploading" } }));

      const fd = new FormData();
      fd.append("participant_id", participantId);
      fd.append("assignment_id", assignmentId);
      fd.append("stimulus_code", stimulusCode);
      fd.append("duration_sec", durationSec.toFixed(2));
      fd.append("attempt_n", String(attemptN));
      fd.append("audio", blob, `${stimulusCode}.webm`);

      try {
        const res = await fetch("/api/pretest-upload", {
          method: "POST",
          body: fd,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error ?? `Server returned ${res.status}`);
        }
        setStates((p) => ({ ...p, [assignmentId]: { status: "saved" } }));
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStates((p) => ({ ...p, [assignmentId]: { status: "error", error: msg } }));
        throw err;
      }
    },
    [participantId],
  );

  const handleFinish = useCallback(async () => {
    setSubmitting(true);
    setDoneError(null);
    // pre-test complete → navigate to pre-test done page.
    // Stage advancement (status field on participant) is left for Phase D.
    router.push(`/pre-test/done?id=${participantId}`);
  }, [participantId, router]);

  return (
    <div className="mt-10 space-y-10">
      {/* Header progress */}
      <div className="flex items-baseline justify-between">
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Pre-test reading · {savedCount} / {total} saved
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">
          Sentence {currentIdx + 1} of {total}
        </p>
      </div>

      {/* Sentence card */}
      <StimulusPrompt
        stimulusCode={current.stimulus_code}
        sentence={current.sentence}
        current={currentIdx + 1}
        total={total}
        targetPhonemes={current.target_phonemes}
        syllableCount={current.syllable_count}
      />

      <AudioRecorder
        key={`rec-${current.assignment_id}`}
        maxDurationSec={PRE_RECORDING_MAX_SEC}
        warnAtSec={PRE_RECORDING_WARN_SEC}
        minDurationSec={PRE_RECORDING_MIN_SEC}
        disabled={submitting}
        onComplete={(blob, dur, attemptN) =>
          handleUpload(current.assignment_id, current.stimulus_code, blob, dur, attemptN)
        }
      />

      {/* Per-sentence navigation */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <button
          type="button"
          onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
          disabled={currentIdx === 0 || submitting}
          className="h-10 px-4 text-sm text-[color:var(--text-muted)] hover:text-[color:var(--text-ink)] disabled:opacity-30"
        >
          ← Previous
        </button>

        {currentIdx < total - 1 ? (
          <button
            type="button"
            onClick={() => setCurrentIdx((i) => Math.min(total - 1, i + 1))}
            disabled={!currentDone || submitting}
            className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--text-ink)] bg-[color:var(--text-ink)] px-6 text-base text-[color:var(--bg-base)] hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Save and continue →
          </button>
        ) : (
          <button
            type="button"
            onClick={handleFinish}
            disabled={!allSaved || submitting}
            className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting
              ? "Finalising…"
              : allSaved
              ? "Finish pre-test"
              : `Finish (${savedCount}/${total} saved)`}
          </button>
        )}
      </div>

      {doneError && (
        <p
          role="alert"
          className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
        >
          {doneError}
        </p>
      )}

      {/* 18-dot progress strip at bottom — academic feel, monospace */}
      <ol className="flex flex-wrap gap-1.5 border-t border-[color:var(--border-warm)] pt-5" aria-label="Pre-test progress">
        {manifest.map((m, i) => {
          const s = states[m.assignment_id]?.status;
          const cls =
            s === "saved"
              ? "bg-[color:var(--success)] border-[color:var(--success)] text-white"
              : s === "error"
              ? "bg-[color:var(--accent-soft)] border-[color:var(--danger)] text-[color:var(--danger)]"
              : i === currentIdx
              ? "bg-[color:var(--accent)] border-[color:var(--accent)] text-white"
              : "bg-[color:var(--bg-elevated)] border-[color:var(--border-warm)] text-[color:var(--text-muted)]";
          return (
            <li key={m.assignment_id}>
              <button
                type="button"
                onClick={() => setCurrentIdx(i)}
                aria-label={`Go to sentence ${i + 1} (${m.stimulus_code})`}
                aria-current={i === currentIdx ? "step" : undefined}
                className={`inline-flex h-7 w-7 items-center justify-center rounded-sm border mono text-[10px] ${cls}`}
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
