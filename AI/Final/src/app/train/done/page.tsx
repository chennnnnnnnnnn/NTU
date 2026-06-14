import { researchServiceClient } from "@/lib/supabase/research";

export default async function TrainDonePage({
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
      .like("storage_path", `${participantId}/train/%`);
    saved = savedC ?? 0;

    const { count: expectedC } = await supa
      .from("research_assignments")
      .select("id", { count: "exact", head: true })
      .eq("participant_id", participantId)
      .eq("test_stage", "train");
    expected = expectedC ?? 0;

    if (saved === expected && expected > 0) {
      await supa
        .from("research_participants")
        .update({ status: "train_done" })
        .eq("id", participantId);
      await supa.from("research_events").insert({
        participant_id: participantId,
        event_type: "session_completed",
        event_payload: { stage: "train", saved, expected },
        client_ts: new Date().toISOString(),
      });
    }
  } catch {
    // best-effort
  }

  const complete = expected > 0 && saved === expected;

  return (
    <Frame>
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Stage 2 of 3 · Training
        </p>
        <h1 className="font-serif">
          {complete ? "Training complete" : "Training almost complete"}
        </h1>
      </header>
      <hr />
      <section className="mt-10 space-y-6">
        <div className="space-y-2">
          <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            Saved
          </p>
          <p className="font-serif text-2xl text-[color:var(--text-ink)]">
            {saved} / {expected} shadowing trials
          </p>
        </div>
        {!complete && (
          <p
            role="alert"
            className="rounded-sm border border-[color:var(--danger)] bg-[color:var(--accent-soft)] p-3 text-sm text-[color:var(--text-ink)]"
          >
            Some trials are missing. Return to{" "}
            <a href={`/train?id=${participantId}`} className="text-[color:var(--accent)] underline-offset-4 hover:underline">
              training
            </a>{" "}
            and finish before continuing.
          </p>
        )}

        <div className="space-y-4 rounded-sm border border-[color:var(--border-warm)] bg-[color:var(--bg-elevated)] p-5">
          <p className="font-serif text-lg">Next: Stage 3 · Post-test</p>
          <p className="text-sm text-[color:var(--text-muted)]">
            You will now read the same 18 sentences once more, again without
            the model audio. This measures how much your reading changed after
            training.
          </p>
          {complete && (
            <a
              href={`/post-test?id=${participantId}`}
              className="inline-flex h-12 items-center justify-center rounded-sm border border-[color:var(--accent)] bg-[color:var(--accent)] px-6 text-base text-white transition-colors hover:bg-[color:var(--accent-hover)]"
            >
              Continue to post-test →
            </a>
          )}
        </div>
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
