import { ConsentForm } from "@/components/ConsentForm";

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ code?: string }>;
}) {
  const params = await searchParams;
  const initialCode = (params.code ?? "").trim().toUpperCase().slice(0, 12);

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-12 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          NTU CSIE7641 · Multimodal HCI · Research Study
        </p>
        <h1 className="font-serif">Shadow Your Perfect Self</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          A user study investigating whether AI-cloned voices that preserve a
          learner&rsquo;s timbre and L1 accent improve English shadowing efficiency
          for Mandarin-L1 speakers.
        </p>
      </header>

      <hr />

      <section className="space-y-6">
        <div className="space-y-2">
          <h2 className="font-serif">Before you begin</h2>
          <p className="max-w-prose text-[color:var(--text-muted)]">
            This session takes about 40 minutes and will record short Mandarin
            and English speech samples. Audio is stored privately, identified
            only by your invitation code, and used solely for academic analysis.
          </p>
        </div>

        <ConsentForm initialCode={initialCode} />
      </section>

      <footer className="mt-auto pt-16">
        <p className="mono text-xs text-[color:var(--text-faint)]">
          Principal investigator: 蔡秉叡 (Ray Tsai) · Department of Computer
          Science and Information Engineering, National Taiwan University ·
          Contact: r14922a21@ntu.edu.tw
        </p>
      </footer>
    </main>
  );
}
