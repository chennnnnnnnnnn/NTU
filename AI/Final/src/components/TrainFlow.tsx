"use client";

import { useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { AudioPlayer } from "@/components/AudioPlayer";
import { AudioRecorder } from "@/components/AudioRecorder";
import { StimulusPrompt } from "@/components/StimulusPrompt";

const TRAIN_MIN_SEC = 2;
const TRAIN_MAX_SEC = 20;
const TRAIN_WARN_SEC = 15;
const TRIALS_PER_STIMULUS = 1; // v1 simplification; v2 may use trials-to-criterion.

type ManifestItem = {
  assignment_id: string;
  stimulus_id: string;
  stimulus_code: string;
  sentence: string;
  syllable_count: number | null;
  target_phonemes: Record<string, number> | null;
  condition: 1 | 2 | 3;
  order_index: number;
  audio_storage_path: string | null;
};

type Props = {
  participantId: string;
  manifest: ManifestItem[];
  /** signed-URL map: stimulus_id → audio URL (may be empty if Phase E not yet run) */
  audioUrls: Record<string, string>;
};

const CONDITION_LABEL: Record<1 | 2 | 3, string> = {
  1: "Native model",
  2: "Self · native accent",
  3: "Self · L1 accent",
};

export function TrainFlow({ participantId, manifest, audioUrls }: Props) {
  const router = useRouter();
  const total = manifest.length;
  const [currentIdx, setCurrentIdx] = useState(0);
  const [listened, setListened] = useState<Record<string, boolean>>({});
  const [listenCounts, setListenCounts] = useState<Record<string, number>>({});
  const [saved, setSaved] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);

  const current = manifest[currentIdx];
  const audioUrl = audioUrls[current.stimulus_id] ?? null;
  const hasListened = Boolean(listened[current.assignment_id]) || audioUrl == null;
  const hasSaved = Boolean(saved[current.assignment_id]);
  const savedCount = useMemo(() => Object.values(saved).filter(Boolean).length, [saved]);
  const allSaved = savedCount === total;

  const handleUpload = useCallback(
    async (blob: Blob, durationSec: number, attemptN: number) => {
      const aid = current.assignment_id;
      const listenCount = listenCounts[aid] ?? 0;
      setErrors((p) => ({ ...p, [aid]: "" }));
      const fd = new FormData();
      fd.append("participant_id", participantId);
      fd.append("assignment_id", aid);
      fd.append("stimulus_code", current.stimulus_code);
      fd.append("test_stage", "train");
      fd.append("trial_n", String(TRIALS_PER_STIMULUS));
      fd.append("duration_sec", durationSec.toFixed(2));
      fd.append("attempt_n", String(attemptN));
      fd.append("listen_count", String(listenCount));
      fd.append("audio", blob, `${current.stimulus_code}_t01.webm`);

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
    [current, participantId, listenCounts],
  );

  const handleFinish = useCallback(() => {
    setSubmitting(true);
    router.push(`/train/done?id=${participantId}`);
  }, [participantId, router]);

  return (
    <div className="mt-10 space-y-10">
      <div className="flex items-baseline justify-between">
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          Training · {savedCount} / {total} saved
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">
          Sentence {currentIdx + 1} of {total} · {CONDITION_LABEL[current.condition]}
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

      <AudioPlayer
        src={audioUrl}
        label={`Model voice · ${CONDITION_LABEL[current.condition]}`}
        initiallyCompleted={Boolean(listened[current.assignment_id])}
        onCompleted={() =>
          setListened((p) => ({ ...p, [current.assignment_id]: true }))
        }
        onPlayStart={(totalListens) =>
          setListenCounts((p) => ({ ...p, [current.assignment_id]: totalListens }))
        }
      />

      {!hasListened && audioUrl != null && (
        <p className="text-sm text-[color:var(--text-muted)]">
          Listen to the model voice all the way through before recording your
          attempt.
        </p>
      )}

      <div className={hasListened ? "" : "pointer-events-none opacity-40"}>
        <AudioRecorder
          key={`rec-${current.assignment_id}`}
          maxDurationSec={TRAIN_MAX_SEC}
          warnAtSec={TRAIN_WARN_SEC}
          minDurationSec={TRAIN_MIN_SEC}
          disabled={!hasListened || submitting}
          onComplete={(b, d, attemptN) => handleUpload(b, d, attemptN)}
        />
      </div>

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
            {allSaved ? "Finish training" : `Finish (${savedCount}/${total})`}
          </button>
        )}
      </div>

      <ol className="flex flex-wrap gap-1.5 border-t border-[color:var(--border-warm)] pt-5" aria-label="Training progress">
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
