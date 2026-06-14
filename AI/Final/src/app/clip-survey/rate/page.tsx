import { ComparisonFlow } from "@/components/survey/ComparisonFlow";
import { getOrCreateSurveyItems } from "@/lib/survey-items";

const ID_RE = /^[0-9a-f-]{36}$/i;

export default async function RatePage({
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
          <a className="underline" href="/clip-survey">起始頁</a>{" "}重新開始。
        </p>
      </main>
    );
  }

  let items;
  try {
    items = await getOrCreateSurveyItems(participantId);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16">
        <h1 className="font-serif">尚無法載入</h1>
        <p className="mt-4 text-[color:var(--text-muted)]">{msg}</p>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-20">
      <header className="space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          步驟 2 / 2 · 語音比較
        </p>
        <h1 className="font-serif">聆聽並比較</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          每題會播放 A、B 兩段語音（皆可重播）。聽完後依題目用 1–5 表達你的判斷，再送出進入下一題。
        </p>
      </header>

      <ComparisonFlow participantId={participantId} items={items} />
    </main>
  );
}
