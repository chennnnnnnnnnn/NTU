import { PreTestDoneFlow } from "@/components/PreTestDoneFlow";
import { researchServiceClient } from "@/lib/supabase/research";

export default async function PreTestDonePage({
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

  // Count saved vs expected, then stamp status if complete.
  let savedCount = 0;
  let expectedCount = 0;
  let statusError: string | null = null;
  try {
    const supa = researchServiceClient();

    const { count: savedC } = await supa
      .from("research_recordings")
      .select("id", { count: "exact", head: true })
      .eq("participant_id", participantId)
      .like("storage_path", `${participantId}/pre/%`);
    savedCount = savedC ?? 0;

    const { count: expectedC } = await supa
      .from("research_assignments")
      .select("id", { count: "exact", head: true })
      .eq("participant_id", participantId)
      .eq("test_stage", "pre");
    expectedCount = expectedC ?? 0;

    if (savedCount === expectedCount && expectedCount > 0) {
      await supa
        .from("research_participants")
        .update({ status: "pre_done" })
        .eq("id", participantId);
      await supa.from("research_events").insert({
        participant_id: participantId,
        event_type: "session_completed",
        event_payload: { stage: "pre", saved: savedCount, expected: expectedCount },
        client_ts: new Date().toISOString(),
      });
    }
  } catch (err) {
    statusError = err instanceof Error ? err.message : String(err);
  }

  const complete = expectedCount > 0 && savedCount === expectedCount;

  return (
    <Frame>
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Stage 1 of 3 · Pre-test
        </p>
        <h1 className="font-serif">
          {complete ? "Pre-test complete" : "Pre-test almost complete"}
        </h1>
      </header>

      <hr />

      <section className="mt-10 space-y-6">
        <div className="space-y-2">
          <p className="mono text-xs uppercase tracking-[0.16em] text-[color:var(--text-muted)]">
            Saved
          </p>
          <p className="font-serif text-2xl text-[color:var(--text-ink)]">
            {savedCount} / {expectedCount} sentences
          </p>
        </div>

        {statusError && (
          <p className="rounded-sm border border-[color:var(--border-strong)] bg-[color:var(--bg-card)] p-3 text-xs text-[color:var(--text-muted)] mono">
            (Status update warning: {statusError})
          </p>
        )}

        <PreTestDoneFlow
          participantId={participantId}
          recordingsSaved={savedCount}
          recordingsExpected={expectedCount}
        />
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
