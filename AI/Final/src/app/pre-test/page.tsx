import { redirect } from "next/navigation";

import { PreTestFlow } from "@/components/PreTestFlow";
import { shouldRedirectFrom } from "@/lib/resume";
import { getOrCreateManifest } from "@/lib/test-assignment";

export default async function PreTestPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const params = await searchParams;
  const participantId = (params.id ?? "").trim();

  if (!/^[0-9a-f-]{36}$/i.test(participantId)) {
    return <MissingId />;
  }

  const redirectTo = await shouldRedirectFrom(participantId, "/pre-test");
  if (redirectTo) redirect(redirectTo);

  try {
    const { manifest } = await getOrCreateManifest(participantId, "pre");

    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
        <header className="mb-10 space-y-3">
          <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
            Stage 1 of 3 · Pre-test
          </p>
          <h1 className="font-serif">Read 18 sentences aloud</h1>
          <p className="max-w-prose text-[color:var(--text-muted)]">
            You will read 18 short English sentences without listening to any
            model first. Try to speak naturally, as if you were saying the line
            in conversation. Each recording lasts 3 to 15 seconds. You can
            re-record any sentence before moving on. There is no right or wrong
            way to read — we are establishing your baseline.
          </p>
        </header>

        <hr />

        <PreTestFlow
          participantId={participantId}
          manifest={manifest.map((m) => ({
            assignment_id: m.assignment_id,
            stimulus_id: m.stimulus_id,
            stimulus_code: m.stimulus_code,
            sentence: m.sentence,
            syllable_count: m.syllable_count,
            target_phonemes: m.target_phonemes,
            order_index: m.order_index,
          }))}
        />
      </main>
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
        <header className="mb-12 space-y-3">
          <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
            Stage 1 of 3 · Pre-test
          </p>
          <h1 className="font-serif">Could not load pre-test</h1>
        </header>
        <p
          role="alert"
          className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
        >
          {msg}
        </p>
      </main>
    );
  }
}

function MissingId() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-12 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Stage 1 of 3 · Pre-test
        </p>
        <h1 className="font-serif">Missing participant id</h1>
      </header>
      <p className="max-w-prose text-[color:var(--text-muted)]">
        This page must be opened after voice processing. Please{" "}
        <a href="/" className="text-[color:var(--accent)] underline-offset-4 hover:underline">
          start from the landing page
        </a>
        .
      </p>
    </main>
  );
}
