import { ProfileForm } from "@/components/survey/ProfileForm";

const ID_RE = /^[0-9a-f-]{36}$/i;

export default async function ProfilePage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const { id } = await searchParams;
  const participantId = (id ?? "").trim();

  if (!ID_RE.test(participantId)) {
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16">
        <h1 className="font-serif">連結無效</h1>
        <p className="mt-4 text-[color:var(--text-muted)]">
          找不到有效的參與編號，請從{" "}
          <a className="underline" href="/clip-survey">
            起始頁
          </a>{" "}
          重新開始。
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          步驟 1 / 2 · 基本資料
        </p>
        <h1 className="font-serif">關於你</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          以下問題協助我們分析不同背景受測者的評分差異，皆為選填。
        </p>
      </header>

      <ProfileForm participantId={participantId} />
    </main>
  );
}
