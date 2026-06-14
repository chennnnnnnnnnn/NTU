import { StartForm } from "@/components/survey/StartForm";
import { TOTAL_QUESTIONS } from "@/lib/survey-config";

export default async function ClipSurveyHome() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 py-16 md:py-24">
      <header className="mb-12 space-y-3">
        <p className="mono text-xs uppercase tracking-[0.22em] text-[color:var(--text-muted)]">
          語音品質評分問卷 · Audio Quality Survey
        </p>
        <h1 className="font-serif">為語音樣本評分</h1>
        <p className="max-w-prose text-[color:var(--text-muted)]">
          這份問卷會請你比較成對的英語語音（共 {TOTAL_QUESTIONS} 題），
          每題聽 A、B 兩段並判斷哪一段在某面向上更好。整份約需 15 分鐘。
        </p>
      </header>

      <hr />

      <section className="space-y-6">
        <div className="space-y-2">
          <h2 className="font-serif">開始之前</h2>
          <p className="max-w-prose text-[color:var(--text-muted)]">
            建議在安靜環境、使用耳機作答以獲得最佳聆聽品質。
            你的填答為匿名，僅以編號識別，且只用於學術分析。
          </p>
        </div>

        <StartForm />
      </section>

      <footer className="mt-auto pt-16">
        <p className="mono text-xs text-[color:var(--text-faint)]">
          NTU CSIE · Multimodal HCI research · 聯絡: r14922a21@ntu.edu.tw
        </p>
      </footer>
    </main>
  );
}
