# Clip-Survey 上線指南（從零開始）

> 目標：把 `/clip-survey` 音檔評分問卷部署成公開網址，受測者在自己裝置完成。
> 你需要：**你自己的** Supabase（免費）+ Vercel（免費）帳號。
> ⚠️ 不要用 shadow 專案的 Supabase / Vercel（那不是你的）。

圖例： 🧑 = 你手動做（網頁/登入）　🤖 = 我可以代跑

---

## A. 建立你自己的 Supabase（🧑）

1. 開 https://supabase.com → **Sign up**（用 GitHub 或 email，免費）
2. 進 Dashboard → **New project**
   - Name：隨意（例如 `clip-survey`）
   - Database Password：設一組並記住
   - Region：選離台灣近的 **Northeast Asia (Tokyo)** 或 **Southeast Asia (Singapore)**
   - 按 Create，等約 1–2 分鐘建好
3. 拿 3 個金鑰：左下 **Project Settings → API**
   - `Project URL`            → 對應 `NEXT_PUBLIC_SUPABASE_URL`
   - `anon` `public` key       → 對應 `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `service_role` `secret` key → 對應 `SUPABASE_SERVICE_ROLE_KEY`（機密！）

---

## B. 建資料表（🧑，2 分鐘）

1. Supabase 左側 **SQL Editor → New query**
2. 打開本專案 `db/migrations/002_survey_tables.sql`，全選複製，貼進去
3. 按 **Run**。看到成功即建好 3 張 `survey_*` 表
   （此檔已自包含，不需要 shadow 的 migration 001）

---

## C. 填 .env.local（🧑 建檔，內容只在本機）

在專案根目錄建立 `.env.local`，填入 A 步驟拿到的值：
```
NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```
> 在對話框輸入 `! ` 開頭可直接建檔，或用你的編輯器。**別把這些值貼進聊天訊息。**

---

## D. 把 18 個音檔寫進 DB（🤖 我可代跑）

```bash
node --env-file=.env.local scripts/seed-clips.mjs
```
成功會印出 18 筆 `alpha1 … gamma6`。

---

## E. 本機測試（🤖 我可代跑 / 🧑 你也可自己跑）

```bash
npm run dev
```
開 http://localhost:3000/clip-survey 走一遍：起始 → 基本資料 → 評分 10 題 → 完成。

---

## F. 部署到 Vercel（🧑 登入 + 🤖 我可代跑指令）

用 Vercel CLI 直接部署這個資料夾（不需要先有 GitHub repo）：

1. 登入（互動，你做）：在對話框輸入
   ```
   ! vercel login
   ```
2. 連結 + 建立新專案（我可代跑）：`vercel link`（選 create new project）
3. 設定 3 個環境變數到 Production（我可代跑，值你提供或從 .env.local 帶）：
   ```
   vercel env add NEXT_PUBLIC_SUPABASE_URL production
   vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY production
   vercel env add SUPABASE_SERVICE_ROLE_KEY production
   ```
4. 正式部署（我可代跑）：
   ```
   vercel --prod
   ```
5. 完成後得到網址，受測者入口：
   ```
   https://<你的專案>.vercel.app/clip-survey
   ```

---

## 之後可選

- 把首頁 `/` 直接導向 `/clip-survey`（你自己的部署可以這樣做）
- 寫 `analyze-survey.mjs` 把 `survey_responses` 匯出成 CSV 分析
- 清掉 repo 內 shadow 相關程式碼，讓專案只剩問卷（較大工程，需要時再做）

---

## 順序總覽

```
A 建 Supabase 帳號+專案(🧑)
  → B 跑 migration 建表(🧑)
  → C 填 .env.local(🧑)
  → D seed 音檔(🤖)
  → E 本機測試(🤖)
  → F Vercel 登入(🧑) + 部署(🤖)
  → 受測者用 /clip-survey 網址
```
