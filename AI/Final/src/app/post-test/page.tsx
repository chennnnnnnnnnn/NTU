import { redirect } from "next/navigation";

import { PostTestFlow } from "@/components/PostTestFlow";
import { shouldRedirectFrom } from "@/lib/resume";
import { getOrCreateManifest } from "@/lib/test-assignment";

export default async function PostTestPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const params = await searchParams;
  const participantId = (params.id ?? "").trim();
  if (!/^[0-9a-f-]{36}$/i.test(participantId)) return <MissingId />;

  const redirectTo = await shouldRedirectFrom(participantId, "/post-test");
  if (redirectTo) redirect(redirectTo);

  try {
    const { manifest } = await getOrCreateManifest(participantId, "post");
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
        <header className="mb-10 space-y-3">
          <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
            Stage 3 of 3 · Post-test
          </p>
          <h1 className="font-serif">Read the same 18 sentences once more</h1>
          <p className="max-w-prose text-[color:var(--text-muted)]">
            This is the same set of sentences from the pre-test. Read them
            naturally, as before — no model voice will play this time. We are
            measuring whether anything changed after training.
          </p>
        </header>
        <hr />
        <PostTestFlow
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
            Stage 3 of 3 · Post-test
          </p>
          <h1 className="font-serif">Could not load post-test</h1>
        </header>
        <p role="alert" className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]">
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
          Stage 3 of 3 · Post-test
        </p>
        <h1 className="font-serif">Missing participant id</h1>
      </header>
      <p className="max-w-prose text-[color:var(--text-muted)]">
        This page must be opened after training. Please{" "}
        <a href="/" className="text-[color:var(--accent)] underline-offset-4 hover:underline">
          start from the landing page
        </a>
        .
      </p>
    </main>
  );
}
