"use client";

import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { AudioRecorder } from "@/components/AudioRecorder";
import { SegmentPrompt } from "@/components/SegmentPrompt";
import {
  CALIBRATION_RECORDING_LIMIT_SEC,
  CALIBRATION_RECORDING_MIN_SEC,
  CALIBRATION_RECORDING_WARN_SEC,
  CALIBRATION_SEGMENTS,
} from "@/lib/calibration-script";

type Status = "recording" | "uploading" | "saved" | "finalizing" | "error";

type SegmentState = {
  status: Status;
  storagePath?: string;
  error?: string;
};

export function CalibrationFlow({ participantId }: { participantId: string }) {
  const router = useRouter();
  const [currentIdx, setCurrentIdx] = useState(0);
  const [states, setStates] = useState<Record<number, SegmentState>>({});
  const [finalizing, setFinalizing] = useState(false);
  const [finalizeError, setFinalizeError] = useState<string | null>(null);

  const totalSegments = CALIBRATION_SEGMENTS.length;
  const currentSegment = CALIBRATION_SEGMENTS[currentIdx];
  const allSaved = useMemo(
    () =>
      CALIBRATION_SEGMENTS.every(
        (seg) => states[seg.id]?.status === "saved",
      ),
    [states],
  );

  const handleUpload = useCallback(
    async (segmentId: number, blob: Blob, durationSec: number, attemptN: number) => {
      setStates((prev) => ({
        ...prev,
        [segmentId]: { status: "uploading" },
      }));

      const fd = new FormData();
      fd.append("participant_id", participantId);
      fd.append("segment_n", String(segmentId));
      fd.append("duration_sec", durationSec.toFixed(2));
      fd.append("attempt_n", String(attemptN));
      fd.append(
        "audio",
        blob,
        `segment-${segmentId}-${Date.now()}.webm`,
      );

      try {
        const res = await fetch("/api/calibration-upload", {
          method: "POST",
          body: fd,
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error ?? `Server returned ${res.status}`);
        }
        const data = (await res.json()) as { storage_path: string };
        setStates((prev) => ({
          ...prev,
          [segmentId]: { status: "saved", storagePath: data.storage_path },
        }));
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStates((prev) => ({
          ...prev,
          [segmentId]: { status: "error", error: msg },
        }));
        throw err;
      }
    },
    [participantId],
  );

  const handleFinalize = useCallback(async () => {
    setFinalizing(true);
    setFinalizeError(null);
    try {
      const res = await fetch("/api/finalize-voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ participant_id: participantId }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Server returned ${res.status}`);
      }
      router.push(`/processing?id=${participantId}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setFinalizeError(msg);
      setFinalizing(false);
    }
  }, [participantId, router]);

  const currentSavedPath = states[currentSegment.id]?.storagePath;
  const currentDone = states[currentSegment.id]?.status === "saved";

  return (
    <div className="mt-10 space-y-10">
      {/* Progress strip */}
      <ol
        className="flex flex-wrap gap-2"
        aria-label="Calibration segments progress"
      >
        {CALIBRATION_SEGMENTS.map((seg, i) => {
          const s = states[seg.id]?.status;
          const dotState =
            s === "saved"
              ? "saved"
              : s === "error"
              ? "error"
              : i === currentIdx
              ? "active"
              : "pending";
          return (
            <li key={seg.id}>
              <button
                type="button"
                onClick={() => setCurrentIdx(i)}
                aria-label={`Go to segment ${seg.id}`}
                aria-current={i === currentIdx ? "step" : undefined}
                className={`inline-flex h-10 w-10 items-center justify-center rounded-sm border mono text-sm transition-colors ${stripStyle(dotState)}`}
              >
                {seg.id}
              </button>
            </li>
          );
        })}
      </ol>

      {/* Current segment */}
      <SegmentPrompt
        segment={currentSegment}
        current={currentIdx + 1}
        total={totalSegments}
      />

      <AudioRecorder
        key={`recorder-${currentSegment.id}`}
        maxDurationSec={CALIBRATION_RECORDING_LIMIT_SEC}
        warnAtSec={CALIBRATION_RECORDING_WARN_SEC}
        minDurationSec={CALIBRATION_RECORDING_MIN_SEC}
        disabled={finalizing}
        onComplete={(blob, dur, attemptN) => handleUpload(currentSegment.id, blob, dur, attemptN)}
      />

      {/* Per-segment navigation */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <button
          type="button"
          onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
          disabled={currentIdx === 0 || finalizing}
          className="h-10 px-4 text-sm text-[color:var(--text-muted)] hover:text-[color:var(--text-ink)] disabled:opacity-30"
        >
          ← Previous
        </button>

        {currentIdx < totalSegments - 1 ? (
          <button
            type="button"
            onClick={() => setCurrentIdx((i) => Math.min(totalSegments - 1, i + 1))}
            disabled={!currentDone || finalizing}
            className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--text-ink)] bg-[color:var(--text-ink)] px-6 text-base text-[color:var(--bg-base)] hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Save and continue →
          </button>
        ) : (
          <button
            type="button"
            onClick={handleFinalize}
            disabled={!allSaved || finalizing}
            className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {finalizing
              ? "Finalising…"
              : allSaved
              ? "Finish calibration"
              : `Finish (${Object.values(states).filter((s) => s.status === "saved").length}/${totalSegments} saved)`}
          </button>
        )}
      </div>

      {finalizeError && (
        <p
          role="alert"
          className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
        >
          {finalizeError}
        </p>
      )}
    </div>
  );
}

function stripStyle(
  s: "saved" | "active" | "pending" | "error",
): string {
  switch (s) {
    case "saved":
      return "border-[color:var(--success)] bg-[color:var(--success)] text-white";
    case "active":
      return "border-[color:var(--accent)] bg-[color:var(--accent)] text-white";
    case "error":
      return "border-[color:var(--danger)] bg-[color:var(--accent-soft)] text-[color:var(--danger)]";
    default:
      return "border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] text-[color:var(--text-muted)]";
  }
}
