"use client";

// Mirror of PreTestFlow but submits to /api/recording-upload with stage=post
// and routes to /post-test/done. Kept as a separate component so the
// research-stage semantics are explicit in the file tree.

import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { AudioRecorder } from "@/components/AudioRecorder";
import { StimulusPrompt } from "@/components/StimulusPrompt";

const POST_RECORDING_MIN_SEC = 3;
const POST_RECORDING_MAX_SEC = 20;
const POST_RECORDING_WARN_SEC = 15;

type ManifestItem = {
  assignment_id: string;
  stimulus_id: string;
  stimulus_code: string;
  sentence: string;
  syllable_count: number | null;
  target_phonemes: Record<string, number> | null;
  order_index: number;
};

type Props = {
  participantId: string;
  manifest: ManifestItem[];
};

export function PostTestFlow({ participantId, manifest }: Props) {
  const router = useRouter();
  const total = manifest.length;
  const [currentIdx, setCurrentIdx] = useState(0);
  const [saved, setSaved] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  const current = manifest[currentIdx];
  const hasSaved = Boolean(saved[current.assignment_id]);
  const savedCount = useMemo(() => Object.values(saved).filter(Boolean).length, [saved]);
  const allSaved = savedCount === total;

  const handleUpload = useCallback(
    async (blob: Blob, durationSec: number, attemptN: number) => {
      const aid = current.assignment_id;
      setErrors((p) => ({ ...p, [aid]: "" }));
      const fd = new FormData();
      fd.append("participant_id", participantId);
      fd.append("assignment_id", aid);
      fd.append("stimulus_code", current.stimulus_code);
      fd.append("test_stage", "post");
      fd.append("duration_sec", durationSec.toFixed(2));
      fd.append("attempt_n", String(attemptN));
      fd.append("audio", blob, `${current.stimulus_code}.webm`);
      try {
        const res = await fetch("/api/recording-upload", { method: "POST", body: fd });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.error ?? `Server returned ${res.status}`);
        }
        setSaved((p) => ({ ...p, [aid]: true }));
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setErrors((p) => ({ ...p, [aid]: msg }));
        throw err;
      }
    },
    [current, participantId],
  );

  const handleFinish = useCallback(() => {
    setSubmitting(true);
    router.push(`/post-test/done?id=${participantId}`);
  }, [participantId, router]);

  return (
    <div className="mt-10 space-y-10">
      <div className="flex items-baseline justify-between">
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Post-test reading · {savedCount} / {total} saved
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">
          Sentence {currentIdx + 1} of {total}
        </p>
      </div>

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
        maxDurationSec={POST_RECORDING_MAX_SEC}
        warnAtSec={POST_RECORDING_WARN_SEC}
        minDurationSec={POST_RECORDING_MIN_SEC}
        disabled={submitting}
        onComplete={(b, d, attemptN) => handleUpload(b, d, attemptN)}
      />

      {errors[current.assignment_id] && (
        <p role="alert" className="text-sm text-[color:var(--danger)]">
          {errors[current.assignment_id]}
        </p>
      )}

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
            disabled={!hasSaved || submitting}
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
            {allSaved ? "Finish post-test" : `Finish (${savedCount}/${total})`}
          </button>
        )}
      </div>

      <ol className="flex flex-wrap gap-1.5 border-t border-[color:var(--border-warm)] pt-5" aria-label="Post-test progress">
        {manifest.map((m, i) => {
          const s = saved[m.assignment_id];
          const cls = s
            ? "bg-[color:var(--success)] border-[color:var(--success)] text-white"
            : i === currentIdx
            ? "bg-[color:var(--accent)] border-[color:var(--accent)] text-white"
            : "bg-[color:var(--bg-elevated)] border-[color:var(--border-warm)] text-[color:var(--text-muted)]";
          return (
            <li key={m.assignment_id}>
              <button
                type="button"
                onClick={() => setCurrentIdx(i)}
                aria-label={`Go to sentence ${i + 1}`}
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
