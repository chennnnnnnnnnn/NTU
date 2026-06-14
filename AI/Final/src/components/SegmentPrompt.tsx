import type { CalibrationSegment } from "@/lib/calibration-script";

type Props = {
  segment: CalibrationSegment;
  current: number;
  total: number;
};

export function SegmentPrompt({ segment, current, total }: Props) {
  return (
    <section className="space-y-5">
      <div className="flex items-baseline justify-between border-b border-[color:var(--border-warm)] pb-3">
        <p className="mono text-xs uppercase tracking-[0.18em] text-[color:var(--text-muted)]">
          Segment {current} of {total}
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">
          {segment.topic} · ~{segment.syllables} syl
        </p>
      </div>

      <div>
        <p className="mb-2 text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)] mono">
          Read aloud in Mandarin
        </p>
        <p
          lang="zh-Hant"
          className="font-serif text-[1.5rem] leading-[1.75] text-[color:var(--text-ink)]"
        >
          {segment.zh}
        </p>
      </div>
    </section>
  );
}
