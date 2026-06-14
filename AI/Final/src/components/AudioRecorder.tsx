"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type RecorderState =
  | "idle"
  | "requesting-permission"
  | "recording"
  | "preview"
  | "uploading"
  | "done"
  | "error";

type Props = {
  /** Soft maximum duration in seconds (counter turns danger color near it). */
  maxDurationSec: number;
  /** Show a danger-colored counter starting at this many seconds elapsed. */
  warnAtSec?: number;
  /** Minimum duration in seconds to allow Finish (prevents accidental empty clips). */
  minDurationSec?: number;
  /**
   * Called with the recorded Blob once the participant confirms the take.
   * attemptN counts how many records preceded this accepted take (1 = first
   * try; 2 = participant re-recorded once; etc.).
   */
  onComplete: (blob: Blob, durationSec: number, attemptN: number) => void | Promise<void>;
  /** Disable while parent is busy (e.g. navigating). */
  disabled?: boolean;
};

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export function AudioRecorder({
  maxDurationSec,
  warnAtSec = Math.max(maxDurationSec - 10, 0),
  minDurationSec = 5,
  onComplete,
  disabled = false,
}: Props) {
  const [state, setState] = useState<RecorderState>("idle");
  const [elapsedSec, setElapsedSec] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTsRef = useRef<number>(0);
  const tickRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const previewBlobRef = useRef<Blob | null>(null);
  const previewDurationRef = useRef<number>(0);
  const attemptCountRef = useRef<number>(0);

  // Waveform refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);

  // ─── Cleanup helpers ───
  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const clearTick = useCallback(() => {
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  const stopWaveform = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    analyserRef.current = null;
  }, []);

  const revokePreview = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    previewBlobRef.current = null;
    previewDurationRef.current = 0;
  }, [previewUrl]);

  // Hard stop on unmount.
  useEffect(() => {
    return () => {
      clearTick();
      stopWaveform();
      stopStream();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Waveform drawing loop ───
  const startWaveform = useCallback((stream: MediaStream) => {
    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioCtx();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);

    audioCtxRef.current = ctx;
    analyserRef.current = analyser;

    const bufferLen = analyser.fftSize;
    const data = new Uint8Array(bufferLen);

    const draw = () => {
      const canvas = canvasRef.current;
      const a = analyserRef.current;
      if (!canvas || !a) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }
      a.getByteTimeDomainData(data);

      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr) canvas.width = w * dpr;
      if (canvas.height !== h * dpr) canvas.height = h * dpr;
      const c = canvas.getContext("2d");
      if (!c) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }
      c.setTransform(dpr, 0, 0, dpr, 0, 0);

      // background
      c.fillStyle = "rgba(242, 237, 228, 0.6)"; // --bg-card with alpha
      c.fillRect(0, 0, w, h);

      // center line
      c.strokeStyle = "rgba(155, 147, 136, 0.4)"; // --text-faint with alpha
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(0, h / 2);
      c.lineTo(w, h / 2);
      c.stroke();

      // waveform
      c.lineWidth = 2;
      c.strokeStyle = "#8B3A2F"; // --accent terracotta
      c.beginPath();
      const sliceWidth = w / bufferLen;
      let x = 0;
      for (let i = 0; i < bufferLen; i++) {
        const v = data[i] / 128.0; // 0..2
        const y = (v * h) / 2;
        if (i === 0) c.moveTo(x, y);
        else c.lineTo(x, y);
        x += sliceWidth;
      }
      c.lineTo(w, h / 2);
      c.stroke();

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
  }, []);

  // ─── Recording lifecycle ───
  const startRecording = useCallback(async () => {
    setErrorMsg(null);
    setElapsedSec(0);
    chunksRef.current = [];
    revokePreview();
    setState("requesting-permission");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
          sampleRate: 48000,
        },
      });
      streamRef.current = stream;

      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "";

      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.start();
      startTsRef.current = Date.now();
      attemptCountRef.current += 1;
      setState("recording");
      startWaveform(stream);

      tickRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTsRef.current) / 1000;
        setElapsedSec(elapsed);
        if (elapsed >= maxDurationSec) {
          recorder.stop();
        }
      }, 200);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const friendly =
        /denied|notallowed/i.test(msg)
          ? "Microphone permission was denied. Open browser settings, allow microphone access for this site, then reload."
          : /notfound|devices/i.test(msg)
          ? "No microphone detected. Connect a mic and reload."
          : `Could not start recording: ${msg}`;
      setErrorMsg(friendly);
      setState("error");
      stopStream();
      stopWaveform();
    }
  }, [maxDurationSec, revokePreview, startWaveform, stopStream, stopWaveform]);

  const finishRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) return;
    clearTick();
    if (recorder.state !== "inactive") {
      recorder.addEventListener(
        "stop",
        () => {
          const finalDuration = (Date.now() - startTsRef.current) / 1000;
          const blob = new Blob(chunksRef.current, {
            type: recorder.mimeType || "audio/webm",
          });
          previewBlobRef.current = blob;
          previewDurationRef.current = finalDuration;
          const url = URL.createObjectURL(blob);
          setPreviewUrl(url);
          stopStream();
          stopWaveform();
          setState("preview");
        },
        { once: true },
      );
      recorder.stop();
    }
  }, [clearTick, stopStream, stopWaveform]);

  const cancelRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    clearTick();
    stopWaveform();
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        /* ignore */
      }
    }
    chunksRef.current = [];
    stopStream();
    setElapsedSec(0);
    setState("idle");
  }, [clearTick, stopStream, stopWaveform]);

  const acceptTake = useCallback(async () => {
    const blob = previewBlobRef.current;
    if (!blob) return;
    setState("uploading");
    try {
      await onComplete(blob, previewDurationRef.current, attemptCountRef.current);
      // Keep the preview URL in "done" state so the user can still listen back
      // (but if parent unmounts/changes key, useEffect cleanup will revoke).
      setState("done");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(`Upload failed: ${msg}`);
      setState("error");
    }
  }, [onComplete]);

  const redoFromPreview = useCallback(() => {
    revokePreview();
    setErrorMsg(null);
    setElapsedSec(0);
    setState("idle");
  }, [revokePreview]);

  const redoFromDone = useCallback(() => {
    revokePreview();
    chunksRef.current = [];
    setErrorMsg(null);
    setElapsedSec(0);
    setState("idle");
  }, [revokePreview]);

  // ─── Computed UI ───
  const overWarn = elapsedSec >= warnAtSec;
  const counterColor = overWarn
    ? "text-[color:var(--danger)]"
    : "text-[color:var(--text-ink)]";
  const meetsMinimum = elapsedSec >= minDurationSec;

  return (
    <div className="flex flex-col gap-4">
      {/* Status row */}
      <div className="flex items-center gap-3">
        <RecordingDot state={state} />
        <span
          className={`mono text-[1.375rem] tabular-nums ${counterColor}`}
          aria-live="polite"
          aria-label="Recording duration"
        >
          {state === "preview" || state === "done" || state === "uploading"
            ? formatTime(previewDurationRef.current)
            : formatTime(elapsedSec)}
          {" / "}
          {formatTime(maxDurationSec)}
        </span>
        {state === "recording" && overWarn && (
          <span className="text-sm text-[color:var(--danger)] mono">
            approaching limit
          </span>
        )}
      </div>

      {/* Waveform — only during recording */}
      {state === "recording" && (
        <canvas
          ref={canvasRef}
          aria-hidden
          className="h-20 w-full rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-card)]"
        />
      )}

      {/* Preview audio — visible in preview / done state */}
      {(state === "preview" || state === "done" || state === "uploading") && previewUrl && (
        <div className="space-y-2 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] p-3">
          <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            {state === "done" ? "Saved take" : "Listen to your take"}
          </p>
          {/* Native <audio> controls are accessible + cross-browser */}
          <audio
            src={previewUrl}
            controls
            preload="metadata"
            className="w-full"
            aria-label="Playback of your recording"
          />
        </div>
      )}

      {/* Controls per state */}
      {(state === "idle" || state === "error") && (
        <button
          type="button"
          onClick={startRecording}
          disabled={disabled}
          style={{
            backgroundColor: "var(--accent)",
            borderColor: "var(--accent)",
            color: "#ffffff",
          }}
          className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border px-6 text-base font-medium transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Start recording"
        >
          <MicIcon />
          {state === "error" ? "Try again" : "Start recording"}
        </button>
      )}

      {state === "requesting-permission" && (
        <p className="text-sm text-[color:var(--text-muted)]">
          Requesting microphone permission…
        </p>
      )}

      {state === "recording" && (
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={finishRecording}
            disabled={!meetsMinimum}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border border-[color:var(--text-ink)] bg-[color:var(--text-ink)] px-6 text-base text-[color:var(--bg-base)] transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Finish recording"
          >
            <StopIcon />
            {meetsMinimum ? "Finish recording" : `Keep going (≥ ${minDurationSec}s)`}
          </button>
          <button
            type="button"
            onClick={cancelRecording}
            className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--border-strong)] bg-transparent px-6 text-base text-[color:var(--text-muted)] hover:text-[color:var(--text-ink)]"
            aria-label="Cancel recording"
          >
            Cancel
          </button>
        </div>
      )}

      {state === "preview" && (
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={acceptTake}
            disabled={disabled}
            style={{
              backgroundColor: "var(--accent)",
              borderColor: "var(--accent)",
              color: "#ffffff",
            }}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border px-6 text-base font-medium transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Use this take"
          >
            <CheckIcon />
            Use this take
          </button>
          <button
            type="button"
            onClick={redoFromPreview}
            disabled={disabled}
            style={{
              borderColor: "var(--border-strong)",
              color: "var(--text-ink)",
            }}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-sm border bg-transparent px-6 text-base hover:opacity-80"
            aria-label="Re-record this segment"
          >
            <MicIcon />
            Re-record
          </button>
        </div>
      )}

      {state === "uploading" && (
        <p className="text-sm text-[color:var(--text-muted)] mono" aria-live="polite">
          Uploading…
        </p>
      )}

      {state === "done" && (
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-2 text-sm text-[color:var(--success)]">
            <CheckIcon />
            Saved ({formatTime(previewDurationRef.current)})
          </span>
          <button
            type="button"
            onClick={redoFromDone}
            disabled={disabled}
            className="text-sm text-[color:var(--accent)] underline-offset-4 hover:underline disabled:opacity-50"
          >
            Re-record this segment
          </button>
        </div>
      )}

      {state === "error" && errorMsg && (
        <div
          role="alert"
          className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
        >
          {errorMsg}
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Visual atoms (SVG only — no emoji per design rubric)
// ──────────────────────────────────────────────────────────────────────

function RecordingDot({ state }: { state: RecorderState }) {
  const base =
    "inline-block h-3 w-3 rounded-full border border-[color:var(--border-strong)]";
  if (state === "recording") {
    return (
      <span
        className={`${base} animate-pulse border-transparent bg-[color:var(--accent)]`}
        aria-hidden
      />
    );
  }
  if (state === "preview") {
    return (
      <span
        className={`${base} border-transparent bg-[color:var(--text-ink)]`}
        aria-hidden
      />
    );
  }
  if (state === "done") {
    return (
      <span
        className={`${base} border-transparent bg-[color:var(--success)]`}
        aria-hidden
      />
    );
  }
  if (state === "uploading") {
    return (
      <span
        className={`${base} animate-pulse border-transparent bg-[color:var(--accent-soft)]`}
        aria-hidden
      />
    );
  }
  if (state === "error") {
    return (
      <span
        className={`${base} border-transparent bg-[color:var(--danger)]`}
        aria-hidden
      />
    );
  }
  return <span className={`${base} bg-[color:var(--bg-elevated)]`} aria-hidden />;
}

function MicIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="9" y="3" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="18" x2="12" y2="21" />
      <line x1="8" y1="21" x2="16" y2="21" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
    >
      <rect x="6" y="6" width="12" height="12" rx="1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}
