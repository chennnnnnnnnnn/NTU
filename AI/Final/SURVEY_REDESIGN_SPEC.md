# Audio-Survey 精簡版 — Design Spec (v0.3 已實作，2026-06-04)

> 由現有 `shadow-research-app`（語音 shadowing 實驗）**精簡改造**而來。
> 目標：一個音檔版問卷網站，受測者填基本資料 → 聽音檔做 Likert 評分 → 收資料。

**v0.3 更新（已實作）**：全部程式碼完成、`npm run build` 通過。
**重要變更（非破壞性）**：因現有 shadow 實驗仍在進行（run 6/1–6/8），未改寫現有
`/`、`/api/enroll`、`/done`，改把新問卷建在獨立路徑 **`/clip-survey/*`** 與
**`/api/survey/*`**，與舊流程完全隔離。

開工前未回答的 open questions 採以下預設（皆可一行改）：
邀請碼**選填**（空白自動產生匿名碼）、評分時**顯示台詞**（`SHOW_SENTENCE=true`）、
音檔放 **`public/clips/`**（公開）。

---

## 0. 已建立的檔案 + 部署步驟（v0.3）

### 新增檔案
```
db/migrations/002_survey_tables.sql        3 張 survey_* 表
scripts/seed-clips.mjs                     掃 public/clips 寫入 survey_clips
src/lib/supabase/survey.ts                 survey_* 白名單 wrapper
src/lib/survey-config.ts                   N、Likert 題目、基本資料欄位（單一改動點）
src/lib/survey-assignment.ts               18 抽 10 + manifest（idempotent）
src/app/api/survey/{enroll,profile,clips,answer}/route.ts
src/app/clip-survey/{page,profile,rate,done}/...
src/components/survey/{StartForm,ProfileForm,RateFlow}.tsx
public/clips/alpha1.mp3 … gamma6.mp3       18 個（已從 stimuli/c1_native 複製）
```

### 部署 / 本機跑（3 步）
```bash
# 1. 跑 migration（在 Supabase SQL editor 貼 db/migrations/002_survey_tables.sql，
#    或用既有連線執行）— 需要 migration 001 已建立的 set_updated_at() 函式
# 2. 把 18 個 clip 寫進 DB
node --env-file=.env.local scripts/seed-clips.mjs
# 3. 本機測試
npm run dev    # 開 http://localhost:3000/clip-survey
```
> `.env.local` 只需 `NEXT_PUBLIC_SUPABASE_URL` / `NEXT_PUBLIC_SUPABASE_ANON_KEY`
> / `SUPABASE_SERVICE_ROLE_KEY`。**不需要** Fish / ElevenLabs 金鑰。

### 受測者入口
`https://<app>/clip-survey`（或附邀請碼 `?code=ABC123`）

---

---

## 1. 目標 (Goal)

讓 ~20 位受測者，先填**基本資料**，再**隨機聽 10 個音檔**（來自 18 個的池子），
每個音檔聽完後做 **Likert 量表評分**，所有答案存進資料庫供後續分析。

一句話：**基本資料 + 音檔版隨機抽樣問卷 + 資料收集後端。**

---

## 2. 範圍 (Scope)

### In（要做）
- 同意/開始頁
- **基本資料頁**（demographics）
- 隨機抽 10 個音檔的指派邏輯（不重複、可斷可續）
- 音檔播放 + Likert 評分介面
- 答案寫入 Supabase
- 完成頁

### Out（明確不做 — 從現有專案砍掉）
- ❌ 聲音克隆（Fish / ElevenLabs API）
- ❌ 受測者錄音（MediaRecorder、所有 upload）
- ❌ Latin square、pre/train/post 三階段、C1/C2/C3 條件
- ❌ 個人化刺激生成

---

## 3. 鎖定決策 (Decisions)

| # | 決策 | 狀態 |
|---|---|---|
| D1 | 每位受測者隨機抽 **10** 個音檔（池子共 18 個） | ✅ 已定 |
| D2 | 評分用 **Likert 量表**（題目見 §7） | ✅ 已定（題目待確認） |
| D3 | **精簡改造**現有 Next.js + Supabase 專案 | ✅ 已定 |
| D4 | 音檔放 **Next.js `public/clips/`**，不走 Supabase Storage | ⚠️ 建議，待確認 |
| D5 | 新表用 `survey_*` 前綴，沿用 service-role 白名單機制 | ⚠️ 建議 |
| D6 | 報名時**一次抽好 10 個並記順序** → 支援斷點續傳、不重複 | ⚠️ 建議 |
| D7 | 受測者識別：匿名 + 邀請碼 `?code=ABC123`（沿用現有模式） | ⚠️ 建議 |
| D8 | **基本資料頁**在同意之後、評分之前 | ✅ 已定（欄位見 §7.1） |
| D9 | 音檔來源 = `stimuli/c1_native/` 頂層 **18 個** mp3（正式 reference set） | ✅ 已定 |

---

## 4. 資料模型 (3 張表)

新表全部用 `survey_` 前綴，與 mirror-app / research_* 隔離。

```sql
-- 1. 受測者（含基本資料）
CREATE TABLE survey_participants (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  external_code text UNIQUE,                 -- 邀請碼，可為匿名 null
  status        text NOT NULL DEFAULT 'enrolled',
                -- enrolled / profile_done / in_progress / completed
  -- 基本資料（§7.1，受測者填，皆可選填）
  age_bracket       text,                    -- "18-24" / "25-34" ...
  gender            text,
  native_language   text,
  english_level     text,                    -- CEFR 自評 A1..C2
  used_headphones   boolean,                 -- 是否戴耳機作答
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);

-- 2. 音檔池（18 筆）
CREATE TABLE survey_clips (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  clip_code    text UNIQUE NOT NULL,          -- "alpha1" .. "gamma6"
  sentence     text,                          -- 對應台詞（README 已有對照）
  storage_path text NOT NULL,                 -- "/clips/alpha1.mp3"
  created_at   timestamptz NOT NULL DEFAULT now()
);

-- 3. 答案（每位受測者 10 列）
CREATE TABLE survey_responses (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  participant_id uuid NOT NULL REFERENCES survey_participants(id) ON DELETE CASCADE,
  clip_id        uuid NOT NULL REFERENCES survey_clips(id),
  order_index    int  NOT NULL,               -- 1..10，抽樣時就定好
  -- Likert 評分（見 §7；先用 4 題 1-5）
  rating_naturalness   int CHECK (rating_naturalness   BETWEEN 1 AND 5),
  rating_clarity       int CHECK (rating_clarity       BETWEEN 1 AND 5),
  rating_nativeness    int CHECK (rating_nativeness    BETWEEN 1 AND 5),
  rating_overall       int CHECK (rating_overall       BETWEEN 1 AND 5),
  listen_count   int DEFAULT 0,               -- 播了幾次（行為資料）
  answered_at    timestamptz,                 -- null = 已指派但還沒答
  created_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (participant_id, clip_id)
);
```

> **抽樣即建列**：填完基本資料後就插入 10 列 `survey_responses`（`answered_at = null`），
> 受測者作答時逐列 `UPDATE`。好處：①斷可續②不重複③重整頁面不會重抽。
> 18 個池子抽 10 個 → 不重複抽樣（每人看到的 10 個各不相同）。

---

## 5. 音檔處理 (D4 + D9)

來源：`stimuli/c1_native/` 頂層 18 個 mp3（`alpha1.mp3`..`gamma6.mp3`，每個 ~5 秒）。

**做法：複製到 `public/clips/`**
- 一行指令把 18 個 mp3 複製進 `public/clips/`
- 前端直接 `<audio src="/clips/alpha1.mp3">`，零設定、CDN 快取、最穩
- `survey_clips.storage_path` 存 `/clips/alpha1.mp3`，`sentence` 由 README 對照表帶入
- 種子腳本 `scripts/seed-clips.mjs` 掃 `public/clips/` + 內建台詞對照 → 寫 18 筆進 DB

> 全部 18 個都是 ElevenLabs「Rachel」native 英語 TTS（見 c1_native/README.md），可公開。
> 若仍不想公開直接抓 URL → 改 Supabase Storage signed URL（較複雜，預設不採）。

---

## 6. 使用者流程 (User Flow)

```
研究員寄出: https://<app>/?code=ABC123

/                同意書 + 邀請碼/開始按鈕
   │  POST /api/enroll → 找到或建立 participant，回 { participant_id, next }
   ↓
/profile?id=...  基本資料頁（§7.1）
   │  POST /api/profile → 存 demographics
   │    → 若還沒指派：18 抽 10，插 10 列 survey_responses
   │    → status = in_progress
   ↓
/survey?id=...   主介面（迴圈 10 次）
   │    每一題：
   │      1. 顯示「第 k / 10 題」
   │      2. <AudioPlayer> 播放音檔（可重播，記 listen_count）
   │      3.（可選）顯示對應台詞 sentence
   │      4. 4 題 Likert（1-5 單選）
   │      5. POST /api/answer → UPDATE 該列
   │      6. 自動載入下一個未答的 clip
   │  10 題全答完 → status = completed
   ↓
/done            謝謝頁（領取資訊等）
```

**斷點續傳**：每個 API 依 status + 「還有沒有未答的列」決定 `next`：
- `enrolled` → `/profile`
- `profile_done` / `in_progress` 且有未答 → `/survey`（從第一個 `answered_at = null` 接續）
- 全部已答 → `/done`

---

## 7. Likert 題目設計（⚠️ 我提的草案，請確認/調整）

這 18 個是 ElevenLabs native 英語 TTS 朗讀的 Friends 句子。採語音評估常見的
**MOS 式 4 題**，全部 **1-5 分**，每題一列單選：

| # | 欄位 | 題目 | 1 分 | 5 分 |
|---|---|---|---|---|
| Q1 | `rating_naturalness` | 這段語音聽起來有多**自然**？ | 非常不自然 | 非常自然 |
| Q2 | `rating_clarity` | 內容有多**清楚易懂**？ | 非常不清楚 | 非常清楚 |
| Q3 | `rating_nativeness` | 聽起來有多像**母語者**？ | 明顯外國口音 | 完全像母語者 |
| Q4 | `rating_overall` | **整體**而言給幾分？ | 很差 | 很好 |

> 這 4 題只是合理預設。若你的研究問題不同（情緒、相似度、性別判斷…），
> 告訴我實際要問什麼，我換掉欄位名與題目即可。

### 7.1 基本資料頁欄位（⚠️ 草案，皆可選填）

| 欄位 | 型態 | 選項範例 |
|---|---|---|
| 年齡層 `age_bracket` | 單選 | 18–24 / 25–34 / 35–44 / 45+ |
| 性別 `gender` | 單選 | 男 / 女 / 不願透露 |
| 母語 `native_language` | 文字或單選 | 中文 / 英文 / 其他 |
| 英語程度 `english_level` | 單選 | CEFR A1 / A2 / B1 / B2 / C1 / C2 |
| 作答時是否戴耳機 `used_headphones` | 是/否 | （音檔評分品質控制常用） |

> 要加/減欄位（例如聽力是否正常、慣用裝置）跟我說。

---

## 8. API 介面

| 端點 | 方法 | Body / Query | 回傳 |
|---|---|---|---|
| `/api/enroll` | POST | `{ code? }` | `{ participant_id, next }` |
| `/api/profile` | POST | `{ participant_id, age_bracket, gender, ... }` | `{ ok, next }`（並完成 18 抽 10） |
| `/api/clips` | GET | `?id=<pid>` | 該受測者的 10 個 clip（含路徑、台詞、order、已答否） |
| `/api/answer` | POST | `{ participant_id, clip_id, ratings{}, listen_count }` | `{ ok, next }` |

全部走 `surveyServiceClient()`（複製現有 `researchServiceClient`，白名單改成 `survey_*`）。

---

## 9. 重用 vs 新建（對照現有專案）

| 現有檔案 | 處置 |
|---|---|
| `src/lib/supabase/*` | ✅ 重用（新增 `survey.ts` 白名單） |
| `src/components/AudioPlayer.tsx` | ✅ 重用 |
| `src/app/globals.css` + `layout.tsx` | ✅ 重用（學術配色） |
| `src/app/page.tsx`（同意頁） | 🔧 簡化重寫 |
| `src/components/AudioRecorder.tsx`、calibration、train… | ❌ 不用（保留在 repo 但不接） |
| `finalize-voice`、`generate-stimuli`、latin-square | ❌ 不用 |

### 新增檔案
```
db/migrations/002_survey_tables.sql      ← 3 張 survey_* 表
src/lib/supabase/survey.ts               ← 白名單 wrapper
src/lib/survey-config.ts                 ← N=10、Likert 題目、基本資料欄位定義
src/app/page.tsx                         ← 改寫：同意 + 開始
src/app/profile/page.tsx                 ← 新增：基本資料頁
src/app/survey/page.tsx                  ← 新增：主問卷介面
src/app/done/page.tsx                    ← 改寫：謝謝頁
src/app/api/enroll/route.ts              ← 改寫：建/找 participant
src/app/api/profile/route.ts             ← 新增：存 demographics + 18 抽 10
src/app/api/clips/route.ts               ← 新增
src/app/api/answer/route.ts              ← 新增
src/components/ProfileForm.tsx           ← 新增：基本資料表單
src/components/SurveyFlow.tsx            ← 新增：播放+評分迴圈
public/clips/alpha1.mp3 ... gamma6.mp3   ← 從 stimuli/c1_native 複製
scripts/seed-clips.mjs                   ← 掃 public/clips 寫 DB
```

---

## 10. ⚠️ 待你確認的 Open Questions

1. **Likert 4 題（§7）** 符合你的研究問題嗎？要不要換題目/欄位？
   （特別注意：18 個全是 native 英語 TTS，Q3「母語程度」可能變化不大 → 是否換題？）
2. **基本資料欄位（§7.1）** 要加/減哪些？
3. **D4 音檔放 `public/`**（任何人有 URL 就能抓）可以嗎？
4. 評分時**要不要顯示對應台詞**（sentence）？顯示 → 偏「清晰度/正確性」判斷；
   不顯示 → 偏「純聽感」判斷。
5. 受測者要**邀請碼**還是純匿名一個按鈕開始？
6. 覆蓋率：18 抽 10、20 人 → 每個音檔約被聽 11 次 ✅（已充足，僅告知）

---

## 11. 建置階段（確認後）

| 階段 | 內容 | 估時 |
|---|---|---|
| P1 | migration 002 + `survey.ts` + `survey-config.ts` + 複製音檔 + seed-clips | 1.5h |
| P2 | enroll + profile（含 18 抽 10）+ clips/answer API | 2.5h |
| P3 | `/survey` 介面（播放 + Likert + 迴圈 + 斷點續傳） | 3h |
| P4 | 同意頁 + 基本資料頁 + 謝謝頁 | 1.5h |
| P5 | 本機測試 + 部署 | 1h |

---

## 下一步

請回覆 §10 的問題（至少 Q1 題目 + Q2 基本資料欄位）。
確認後我就從 P1 開始實作。
