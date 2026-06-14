// Display one English stimulus sentence + phoneme target hints.
// Server component (no hooks).

type PhonemeMap = Record<string, number>;

type Props = {
  stimulusCode: string;
  sentence: string;
  current: number;
  total: number;
  targetPhonemes?: PhonemeMap | null;
  syllableCount?: number | null;
};

const PHONEME_LABELS: Record<string, string> = {
  theta_eth: "/θ ð/",
  v: "/v/",
  ae_eh: "/æ ɛ/",
  i_ih: "/i ɪ/",
  n_ng_final: "n/ŋ-final",
};

export function StimulusPrompt({
  stimulusCode,
  sentence,
  current,
  total,
  targetPhonemes,
  syllableCount,
}: Props) {
  return (
    <section className="space-y-5">
      <div className="flex items-baseline justify-between border-b border-[color:var(--border-warm)] pb-3">
        <p className="mono text-xs uppercase tracking-[0.18em] text-[color:var(--text-muted)]">
          Sentence {current} of {total}
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">
          {stimulusCode}
          {syllableCount != null ? ` · ${syllableCount} syl` : ""}
        </p>
      </div>

      <div>
        <p className="mb-2 text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)] mono">
          Read aloud once
        </p>
        <p
          lang="en"
          className="font-serif text-[1.625rem] leading-[1.4] text-[color:var(--text-ink)]"
        >
          {sentence}
        </p>
      </div>

      {targetPhonemes && (
        <div className="space-y-1 border-t border-[color:var(--border-warm)] pt-3">
          <p className="mono text-[10px] uppercase tracking-[0.16em] text-[color:var(--text-faint)]">
            Listen for these contrasts as you read
          </p>
          <ul className="flex flex-wrap gap-x-4 gap-y-1 mono text-xs text-[color:var(--text-muted)]">
            {Object.entries(targetPhonemes)
              .filter(([, n]) => n > 0)
              .map(([key, n]) => (
                <li key={key}>
                  <span className="text-[color:var(--text-ink)]">
                    {PHONEME_LABELS[key] ?? key}
                  </span>{" "}
                  ×{n}
                </li>
              ))}
          </ul>
        </div>
      )}
    </section>
  );
}
