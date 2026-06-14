"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  /** signed URL or public URL of the model audio */
  src: string | null;
  /** Optional caption shown above the play button (e.g. "Model voice · C2") */
  label?: string;
  /** Called once the audio has been listened to in its entirety. */
  onCompleted?: () => void;
  /** Called every time the participant clicks Play (1 = first listen, 2+ = listen again). */
  onPlayStart?: (totalListens: number) => void;
  /** Whether the participant has already played + finished this audio in this session. */
  initiallyCompleted?: boolean;
};

type PlayerState =
  | "idle"
  | "playing"
  | "ended"
  | "error";

export function AudioPlayer({ src, label, onCompleted, onPlayStart, initiallyCompleted }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [state, setState] = useState<PlayerState>(initiallyCompleted ? "ended" : "idle");
  const [currentSec, setCurrentSec] = useState(0);
  const [durationSec, setDurationSec] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const completedRef = useRef(Boolean(initiallyCompleted));
  const listenCountRef = useRef(0);

  useEffect(() => {
    // reset when src changes
    completedRef.current = false;
    listenCountRef.current = 0;
    setCurrentSec(0);
    setDurationSec(null);
    setErrorMsg(null);
    setState("idle");
  }, [src]);

  const handlePlay = useCallback(async () => {
    const a = audioRef.current;
    if (!a || !src) return;
    setErrorMsg(null);
    try {
      await a.play();
      listenCountRef.current += 1;
      onPlayStart?.(listenCountRef.current);
      setState("playing");
    } catch (err) {
      // play() rejects with AbortError when interrupted by a pause() or a new
      // load (e.g. rapid clicks / replays). That's not a real failure — don't
      // get stuck in the error state; just return to idle so it can play again.
      if (err instanceof DOMException && err.name === "AbortError") {
        setState("idle");
        return;
      }
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(`Playback failed: ${msg}`);
      setState("error");
    }
  }, [src, onPlayStart]);

  const handlePause = useCallback(() => {
    audioRef.current?.pause();
    setState("idle");
  }, []);

  const handleReplay = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = 0;
    void handlePlay();
  }, [handlePlay]);

  if (!src) {
    return (
      <div className="rounded-sm border border-dashed border-[color:var(--border-strong)] bg-[color:var(--bg-card)] p-4 text-sm text-[color:var(--text-muted)]">
        Audio is not available yet for this stimulus. Please notify the
        researcher; skip this item if necessary.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {label && (
        <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
          {label}
        </p>
      )}

      <audio
        ref={audioRef}
        src={src}
        preload="auto"
        onLoadedMetadata={(e) => {
          const d = (e.target as HTMLAudioElement).duration;
          if (Number.isFinite(d)) setDurationSec(d);
        }}
        onTimeUpdate={(e) => setCurrentSec((e.target as HTMLAudioElement).currentTime)}
        onEnded={() => {
          setState("ended");
          if (!completedRef.current) {
            completedRef.current = true;
            onCompleted?.();
          }
        }}
        onError={() => {
          setErrorMsg("Audio file failed to load.");
          setState("error");
        }}
      />

      <div className="flex items-center gap-4">
        {state === "idle" && (
          <button
            type="button"
            onClick={handlePlay}
            style={{ borderColor: "var(--text-ink)", color: "var(--text-ink)" }}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border bg-transparent px-6 text-base font-medium hover:bg-[color:var(--bg-card)]"
            aria-label="Play model audio"
          >
            <PlayIcon />
            {completedRef.current ? "Play again" : "Play model voice"}
          </button>
        )}
        {state === "playing" && (
          <button
            type="button"
            onClick={handlePause}
            style={{ borderColor: "var(--text-ink)", color: "var(--text-ink)" }}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border bg-transparent px-6 text-base font-medium hover:bg-[color:var(--bg-card)]"
            aria-label="Pause model audio"
          >
            <PauseIcon />
            Pause
          </button>
        )}
        {state === "ended" && (
          <button
            type="button"
            onClick={handleReplay}
            style={{ borderColor: "var(--text-ink)", color: "var(--text-ink)" }}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border bg-transparent px-6 text-base font-medium hover:bg-[color:var(--bg-card)]"
            aria-label="Replay model audio"
          >
            <PlayIcon />
            Listen again
          </button>
        )}
        {state === "error" && (
          <button
            type="button"
            onClick={handlePlay}
            style={{ borderColor: "var(--danger)", color: "var(--danger)" }}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border bg-transparent px-6 text-base font-medium"
          >
            Retry
          </button>
        )}

        <span className="mono text-sm tabular-nums text-[color:var(--text-muted)]">
          {format(currentSec)}
          {durationSec != null ? ` / ${format(durationSec)}` : ""}
        </span>
      </div>

      {errorMsg && (
        <p role="alert" className="text-sm text-[color:var(--danger)]">
          {errorMsg}
        </p>
      )}
    </div>
  );
}

function format(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${String(Math.floor(s / 60)).padStart(1, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <polygon points="6 4 20 12 6 20" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <rect x="6" y="5" width="4" height="14" />
      <rect x="14" y="5" width="4" height="14" />
    </svg>
  );
}
