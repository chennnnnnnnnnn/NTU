export default async function DoneCompletePage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const params = await searchParams;
  const participantId = (params.id ?? "").trim();
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Complete
        </p>
        <h1 className="font-serif">Thank you</h1>
      </header>
      <hr />
      <section className="mt-10 space-y-6">
        <p className="text-[color:var(--text-ink)]">
          Your participation in the study is complete. The researcher will be
          in touch about the compensation outlined in your invitation email.
        </p>
        <p className="text-[color:var(--text-muted)]">
          You may close this tab. If you experienced any technical problem
          during the session, please mention the participant id below so we
          can investigate.
        </p>
        <p className="mono text-xs text-[color:var(--text-faint)]">
          participant_id · {participantId || "—"}
        </p>
      </section>
    </main>
  );
}
