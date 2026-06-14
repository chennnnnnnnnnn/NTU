# Clip-Survey — 語音 A/B 比較問卷

線上感知評估（perceptual A/B test）：以人耳盲測，比較三種「本人音色合成語音」在**口音、自然度、真人感**上的差異。

- **正式網址**：https://clip-survey.vercel.app/clip-survey
- **完整實驗說明**：見 [`CLIP_SURVEY_OVERVIEW.md`](./CLIP_SURVEY_OVERVIEW.md)
- **設計文件**：見 [`SURVEY_REDESIGN_SPEC.md`](./SURVEY_REDESIGN_SPEC.md)

---

## 這是什麼

受試者聽成對的語音（同一句、同一 speaker、不同合成條件），用 1–5 量表判斷哪段更好：

| 題型 | 配對 | 問題 |
|---|---|---|
| Q1 口音 | c2 vs c3b | 哪段更像母語者？ |
| Q2 真人 | c3a vs c3b | 哪段更像真人說話？ |
| Q3 自然 | c2 vs c3a | 哪段更自然？ |

條件：`c2` 本人音色+母語腔、`c3a` 本人音色+中式腔(Fish)、`c3b` 本人音色+中式腔(ElevenLabs)。
素材：6 speaker × 18 句 × 3 條件 = 324 個音檔，放在 `public/clips/<學號>/`。

## 技術棧

- **Next.js 16** (App Router) + React 19 + TypeScript
- **Tailwind CSS v4**
- **Supabase**（Postgres 資料表 + 靜態音檔由 Next.js `public/` 提供）
- 部署：**Vercel**

## 專案結構（重點）

```
src/app/clip-survey/        起始/基本資料/比較/完成 四頁
src/app/api/survey/         enroll / profile / clips / answer 四個 API
src/components/survey/      StartForm / ProfileForm / ComparisonFlow
src/lib/survey-config.ts    抽樣、題型、量表、選項設定（單一改動點）
src/lib/survey-items.ts     每位受試者的隨機題目生成邏輯
db/migrations/              001~003 資料表 SQL
public/clips/<學號>/        324 個音檔
output/analyze.mjs          結果分析腳本
```

## 資料表（Supabase，`survey_*`）

| 表 | 內容 |
|---|---|
| `survey_participants` | 受試者：姓名、第幾人、背景、狀態、attention_passed |
| `survey_clips` | 324 音檔（學號 / 句組 / 句子 / 條件） |
| `survey_items` | 每題一列：題型、A/B 條件、A/B 順序、1–5 評分 |

## 本機開發

```bash
# 1. 安裝依賴（見 requirements.txt；需 Node >= 22）
npm install

# 2. 設定環境變數
cp .env.local.example .env.local   # 然後填入 Supabase 三把金鑰

# 3. 建資料表：把 db/migrations/001~003 貼進 Supabase SQL Editor 執行
#    （ALTER 後若 API 報「找不到欄位」，跑 NOTIFY pgrst, 'reload schema';）

# 4. 把音檔登錄進 survey_clips
node --env-file=.env.local scripts/seed-clips.mjs

# 5. 啟動
npm run dev   # http://localhost:3000/clip-survey
```

> 注意：`scripts/seed-clips.mjs` 與 `output/analyze.mjs` 需 **Node ≥ 22**（Node 20 缺原生 WebSocket，supabase-js 會報錯）。

## 部署（Vercel）

```bash
npx vercel link
npx vercel env add NEXT_PUBLIC_SUPABASE_URL production
npx vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY production
npx vercel env add SUPABASE_SERVICE_ROLE_KEY production
npx vercel --prod
```

> Git 提交者 email 必須對應到 Vercel 帳號的 GitHub，否則部署會被擋（"commit email could not be matched"）。

## 分析結果

```bash
# 從 Supabase 匯出三張表 CSV 到 output/，然後：
node output/analyze.mjs
```
輸出各題型偏好計票、sign test、以及 α/β/γ 分層結果。
