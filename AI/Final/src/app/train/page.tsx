import { redirect } from "next/navigation";

import { TrainFlow } from "@/components/TrainFlow";
import { shouldRedirectFrom } from "@/lib/resume";
import { researchServiceClient } from "@/lib/supabase/research";
import { getOrCreateManifest } from "@/lib/test-assignment";

export default async function TrainPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const params = await searchParams;
  const participantId = (params.id ?? "").trim();
  if (!/^[0-9a-f-]{36}$/i.test(participantId)) return <MissingId />;

  const redirectTo = await shouldRedirectFrom(participantId, "/train");
  if (redirectTo) redirect(redirectTo);

  try {
    const { manifest } = await getOrCreateManifest(participantId, "train");

    // Try to mint signed URLs for any stimuli that have audio paths.
    // If Phase E hasn't generated audio yet, the URL map will be sparse.
    const supa = researchServiceClient();
    const paths = manifest
      .filter((m) => m.audio_storage_path)
      .map((m) => m.audio_storage_path as string);

    const audioUrls: Record<string, string> = {};
    if (paths.length > 0) {
      const { data, error } = await supa.storage
        .from("research_stimuli")
        .createSignedUrls(paths, 60 * 60); // 1 hour
      if (!error && data) {
        for (let i = 0; i < manifest.length; i++) {
          const m = manifest[i];
          if (m.audio_storage_path) {
            const item = data.find((d) => d.path === m.audio_storage_path);
            if (item?.signedUrl) audioUrls[m.stimulus_id] = item.signedUrl;
          }
        }
      }
    }

    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
        <header className="mb-10 space-y-3">
          <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
            Stage 2 of 3 · Training
          </p>
          <h1 className="font-serif">Shadow each sentence</h1>
          <p className="max-w-prose text-[color:var(--text-muted)]">
            For each of the 18 sentences you will hear one model voice. Listen
            once, then try to repeat the sentence with the same rhythm and
            intonation. Match the model as closely as you can; small mistakes
            are fine.
          </p>
        </header>

        <hr />

        <TrainFlow
          participantId={participantId}
          audioUrls={audioUrls}
          manifest={manifest.map((m) => ({
            assignment_id: m.assignment_id,
            stimulus_id: m.stimulus_id,
            stimulus_code: m.stimulus_code,
            sentence: m.sentence,
            syllable_count: m.syllable_count,
            target_phonemes: m.target_phonemes,
            condition: m.condition,
            order_index: m.order_index,
            audio_storage_path: m.audio_storage_path,
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
            Stage 2 of 3 · Training
          </p>
          <h1 className="font-serif">Could not load training</h1>
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
          Stage 2 of 3 · Training
        </p>
        <h1 className="font-serif">Missing participant id</h1>
      </header>
      <p className="max-w-prose text-[color:var(--text-muted)]">
        This page must be opened after the pre-test stage. Please{" "}
        <a href="/" className="text-[color:var(--accent)] underline-offset-4 hover:underline">
          start from the landing page
        </a>
        .
      </p>
    </main>
  );
}
