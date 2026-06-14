import { ProcessingStatus } from "@/components/ProcessingStatus";

export default async function ProcessingPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const params = await searchParams;
  const participantId = (params.id ?? "").trim();

  if (!/^[0-9a-f-]{36}$/i.test(participantId)) {
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
        <header className="mb-12 space-y-3">
          <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
            Step 3 of 3 · Voice processing
          </p>
          <h1 className="font-serif">Missing participant id</h1>
        </header>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          This page must be opened after calibration. Please{" "}
          <a href="/" className="text-[color:var(--accent)] underline-offset-4 hover:underline">
            start from the beginning
          </a>
          .
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Step 3 of 3 · Voice processing
        </p>
        <h1 className="font-serif">Cloning your voice</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          Two services are now creating a model from your Mandarin recordings.
          This usually finishes within one minute. You can leave this tab open
          and stretch.
        </p>
      </header>

      <hr />

      <ProcessingStatus participantId={participantId} />
    </main>
  );
}
