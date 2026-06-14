export default async function DonePage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          已完成 · Complete
        </p>
        <h1 className="font-serif">謝謝你的參與</h1>
      </header>

      <hr />

      <section className="space-y-4">
        <p className="max-w-prose text-[color:var(--text-muted)]">
          你已完成所有語音評分，填答已安全儲存。可以關閉此頁面了。
        </p>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          若有任何問題，歡迎聯絡研究人員。
        </p>
      </section>

      <footer className="mt-auto pt-16">
        <p className="mono text-xs text-[color:var(--text-faint)]">
          NTU CSIE · Multimodal HCI research · 聯絡: r14922a21@ntu.edu.tw
        </p>
      </footer>
    </main>
  );
}
