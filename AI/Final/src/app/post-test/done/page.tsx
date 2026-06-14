import { researchServiceClient } from "@/lib/supabase/research";

export default async function PostTestDonePage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const params = await searchParams;
  const participantId = (params.id ?? "").trim();
  if (!/^[0-9a-f-]{36}$/i.test(participantId)) {
    return (
      <Frame>
        <h1 className="font-serif">Missing participant id</h1>
      </Frame>
    );
  }

  let saved = 0;
  let expected = 0;
  try {
    const supa = researchServiceClient();
    const { count: savedC } = await supa
      .from("research_recordings")
      .select("id", { count: "exact", head: true })
      .eq("participant_id", participantId)
      .like("storage_path", `${participantId}/post/%`);
    saved = savedC ?? 0;
    const { count: expectedC } = await supa
      .from("research_assignments")
      .select("id", { count: "exact", head: true })
      .eq("participant_id", participantId)
      .eq("test_stage", "post");
    expected = expectedC ?? 0;

    if (saved === expected && expected > 0) {
      await supa
        .from("research_participants")
        .update({ status: "post_done" })
        .eq("id", participantId);
      await supa.from("research_events").insert({
        participant_id: participantId,
        event_type: "session_completed",
        event_payload: { stage: "post", saved, expected },
        client_ts: new Date().toISOString(),
      });
    }
  } catch {
    /* best-effort */
  }
  const complete = expected > 0 && saved === expected;

  return (
    <Frame>
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Stage 3 of 3 · Post-test
        </p>
        <h1 className="font-serif">
          {complete ? "Post-test complete" : "Post-test almost complete"}
        </h1>
      </header>
      <hr />
      <section className="mt-10 space-y-6">
        <div className="space-y-2">
          <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            Saved
          </p>
          <p className="font-serif text-2xl text-[color:var(--text-ink)]">
            {saved} / {expected} sentences
          </p>
        </div>
        {complete ? (
          <div className="space-y-4 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] p-5">
            <p className="font-serif text-lg">One last thing</p>
            <p className="text-sm text-[color:var(--text-muted)]">
              Please answer a short survey about your experience. It takes
              about two minutes.
            </p>
            <a
              href={`/done?id=${participantId}`}
              className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)]"
            >
              Continue to survey →
            </a>
          </div>
        ) : (
          <p role="alert" className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]">
            Some sentences are missing recordings. Return to{" "}
            <a href={`/post-test?id=${participantId}`} className="text-[color:var(--accent)] underline-offset-4 hover:underline">
              post-test
            </a>{" "}
            and finish before continuing.
          </p>
        )}
      </section>
    </Frame>
  );
}

function Frame({ children }: { children: React.ReactNode }) {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      {children}
    </main>
  );
}
