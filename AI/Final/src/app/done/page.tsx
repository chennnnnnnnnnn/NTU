import { SurveyForm } from "@/components/SurveyForm";

export default async function DonePage({
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
            Final · Survey
          </p>
          <h1 className="font-serif">Missing participant id</h1>
        </header>
      </main>
    );
  }
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Final · Survey
        </p>
        <h1 className="font-serif">Tell us how it felt</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          Six short statements followed by a ranking of the three voice
          conditions. There are no right answers; first impressions are fine.
        </p>
      </header>
      <hr />
      <div className="mt-10">
        <SurveyForm participantId={participantId} />
      </div>
    </main>
  );
}
