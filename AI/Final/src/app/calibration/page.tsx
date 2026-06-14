import { redirect } from "next/navigation";

import { CalibrationFlow } from "@/components/CalibrationFlow";
import { shouldRedirectFrom } from "@/lib/resume";

export default async function CalibrationPage({
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
            Step 2 of 3 · Calibration
          </p>
          <h1 className="font-serif">Missing participant id</h1>
        </header>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          This page must be opened from the consent step. Please return to the{" "}
          <a href="/" className="text-[color:var(--accent)] underline-offset-4 hover:underline">
            landing page
          </a>{" "}
          and submit your invitation code first.
        </p>
      </main>
    );
  }

  // Resume guard: if the participant has already finished calibration, jump
  // them straight to where they should be.
  const redirectTo = await shouldRedirectFrom(participantId, "/calibration");
  if (redirectTo) redirect(redirectTo);

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-10 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          Step 2 of 3 · Calibration
        </p>
        <h1 className="font-serif">Record your voice</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          You will read five short Mandarin sentences aloud. Each takes about 30
          to 60 seconds. Speak in a natural conversational tone, as if you were
          telling a friend. Find a quiet space and place the microphone close
          but not touching your mouth.
        </p>
      </header>

      <hr />

      <CalibrationFlow participantId={participantId} />
    </main>
  );
}
