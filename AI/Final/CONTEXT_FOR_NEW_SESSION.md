# Context for new Claude Code session inside research-website/

> 此檔案讓你 `cd research-website/ && claude` 開新 session 時，5 分鐘銜接全部脈絡。
> Last updated: 2026-05-24 by previous session in MHCI final_project/

---

## 🚀 開新 session 第一句話

```
請讀 ./DESIGN.md 和 ./CONTEXT_FOR_NEW_SESSION.md, 我們要繼續建 research-website
(MHCI final project Part 3 的 user study web app)
```

---

## 一、你是誰、要做什麼

- **學生**: 蔡秉叡 (Ray-Rick) · r14922a21
- **課程**: NTU CSIE7641 Multimodal HCI
- **大目標**: Shadow Your Perfect Self 研究 (Part 2 paper 已 5/26 截止 / Part 3 6/9 截止)
- **本資料夾任務**: 建一個 Next.js web app, 讓 N=20 受試者在家完成 shadowing 實驗
- **GitHub**: github.com/rick-ray-wldd/shadow-research-app (private)
- **Deploy 目標**: Vercel (你已登入 allcarerickray-7044)

---

## 二、5 個鎖定決策（D1-D5）

| # | Decision | 含義 |
|---|---|---|
| D1 | Hybrid backend, wrap mirror-app | 重用 mirror-app 的 Fish TTS / calibration question endpoints |
| D2 | Anonymous + external_code | 一人一個邀請 URL `?code=ABC123`, 不做 email auth |
| D3 | Pre-generated stimuli | 本機跑 batch script 產 738 mp3 上 Storage, web 不接 ElevenLabs/Fish API |
| D4 | Client-direct signed-URL upload | 受試者錄音直傳 Supabase Storage, backend 不中轉 |
| D5 | Voice cloning offloaded to mirror-app | 受試者先用 mirror-app onboarding 拿 Fish voice_id, 你研究員額外為他建 ElevenLabs voice_id |

→ research-website 縮成「**stimuli player + recording uploader + event logger**」

---

## 三、5 個 screen (受試者 user flow)

```
/              consent + 邀請碼驗證
/check-voice   確認 fish/elevenlabs voice_id 已綁定
/pre-test      cold-read 18 句, 沒 audio model
/train         Latin-square 配 stimulus → 播 model audio → 受試者 shadow
/post-test     cold-read 18 句再一次
/delayed-post  +7 天後同 post-test (URL 同個)
/done          Likert + 偏好排序 + 領取 7-11 卡 NT$200 資訊
```

---

## 四、Tech stack (鎖定)

```
Next.js 15 App Router
TypeScript
Tailwind + shadcn/ui
MediaRecorder API (browser native)
Supabase (Postgres + Storage)  ← 跟 mirror-app 同一 project
Vercel deploy
GitHub Actions (optional, CI)
```

---

## 五、目前已寫好的東西 (這個 session)

```
research-website/
├── DESIGN.md                            ✅ 完整設計文件
├── CONTEXT_FOR_NEW_SESSION.md           ✅ 本檔案
└── db/migrations/
    └── 001_research_tables.sql          ✅ 5 個 research_* tables + RLS
```

---

## 六、下一個 session 該做什麼 (5/27 二, Part 2 交完之後)

### Day 0 setup (~2 hr)
- [ ] `cd research-website/`
- [ ] `pnpm create next-app@latest . --typescript --tailwind --app --src-dir=false --use-pnpm`
- [ ] `pnpm dlx shadcn@latest init`
- [ ] 安裝 shadcn 元件: `button card input progress dialog form sonner`
- [ ] `pnpm add @supabase/supabase-js @supabase/ssr`
- [ ] 連到 mirror-app 同一個 Supabase project (拿 URL + anon key + service_role key)
- [ ] 跑 migration: `psql ... -f db/migrations/001_research_tables.sql`
- [ ] 在 Supabase Dashboard 建 2 個 bucket (research_stimuli 公開讀 / research_recordings 私有)
- [ ] `.env.local` 填好
- [ ] `git init && git remote add origin git@github.com:rick-ray-wldd/shadow-research-app.git`
- [ ] First commit + push

### Day 1 (5/28 三) ── consent + voice-check (~6 hr)
- [ ] Layout + 全站 design (shadcn theme)
- [ ] Screen 1: `/` 邀請碼驗證 + consent
- [ ] Screen 2: `/check-voice` voice_id 確認
- [ ] `/api/enroll` route handler
- [ ] `/api/manifest` route handler (回 stimuli list)
- [ ] components: `<ConsentForm />`, `<VoiceStatus />`

### Day 2 (5/29 四) ── pre-test recording (~6 hr)
- [ ] `<AudioRecorder />` component (MediaRecorder)
- [ ] `<AudioPlayer />` component
- [ ] Screen 3: `/pre-test` (18 句 cold read 流程)
- [ ] `/api/upload-url` signed URL backend
- [ ] Client-direct upload to Storage
- [ ] research_recordings insert

### Day 3 (5/30 五) ── train + post-test (~6 hr)
- [ ] Latin-square 邏輯 in `lib/stimuli-config.ts`
- [ ] Screen 4: `/train` (含 model audio playback + shadow recording loop)
- [ ] Screen 5: `/post-test` (沿用 pre-test 元件)
- [ ] Screen 6: `/done` Likert + ranking

### Day 4 (5/31 六) ── stimuli + deploy + audit (~5 hr)
- [ ] 本機跑 batch script 產 738 mp3
- [ ] 上傳 Storage (script: `scripts/upload-stimuli.ts`)
- [ ] 跑 `/codex-review-research-app api-routes` audit
- [ ] 修 critical bug
- [ ] Vercel deploy production
- [ ] 自己 dress rehearsal 1 次

### Day 5+ (6/1-6/8) ── run study
- [ ] 寄邀請信 N=20
- [ ] 監控 Supabase Dashboard 看 data 進來
- [ ] 期間用 sister skill `/codex-review-research-app security` + `/codex-review-research-app audio-pipeline` review

---

## 七、可用的 helper skills (我們上個 session 建好的)

### `codex-audit-mirror-app`
查 mirror-app 的東西，read-only，不會動到 production。

```
/codex-audit-mirror-app calibration-flow
/codex-audit-mirror-app tts-pipeline
/codex-audit-mirror-app voice-storage
/codex-audit-mirror-app api-surface
/codex-audit-mirror-app db-schema
/codex-audit-mirror-app <自訂 prompt>
```

### `codex-review-research-app`
review 我們新建的 research-website，找 bug + 安全問題。

```
/codex-review-research-app api-routes
/codex-review-research-app security
/codex-review-research-app schema-consistency
/codex-review-research-app ux-flow
/codex-review-research-app audio-pipeline
/codex-review-research-app <自訂 prompt>
```

報告自動存到 `research-website/.review/YYYY-MM-DD_HHMMSS_<scope>.md`

---

## 八、相關外部資源 (reference)

| 資源 | 位置 |
|---|---|
| Codex 完整 audit report | `~/Documents/NTU/NTUCS_AI/HCI LAB/office hour/2026-05-24_codex-audit_comprehensive-build-prep.md` |
| Mirror-app (READ-ONLY, 不動) | `~/Documents/NTU/NTUCS_AI/2026_Summer_Programs/2026NTUAI_Builders_Challenge/instabrain/mirror-app/` |
| Mirror-app backend | `~/.../instabrain/instabrain-backend/` |
| Part 1 pitch (已交) | `~/.../MHCI/final_project/Shadow-Your-Perfect-Self.pdf` |
| Part 2 Intro v2 | `~/.../MHCI/final_project/part2/introduction_draft_v2.md` |
| 18 stimuli | `~/.../MHCI/final_project/stimuli/stimuli_18_friends.md` |
| Figure 1 SVG | `~/.../MHCI/final_project/figures/Figure1_pipeline.svg` |
| Batch generation script | `~/.../MHCI/final_project/scripts/batch_generate_stimuli.py` |
| PPG-DTW evaluator | `~/.../MHCI/final_project/scripts/evaluate_ppg_dtw.py` |

---

## 九、ENV variables 你需要的

```bash
# Supabase (same as mirror-app project)
NEXT_PUBLIC_SUPABASE_URL=https://<...>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=ey...
SUPABASE_SERVICE_ROLE_KEY=ey...       # server-only

# Optional: if research-website ever needs to call ElevenLabs directly
# (D3 暫定不需要, 預生成完成上 Storage 即可)
ELEVENLABS_API_KEY=sk_...

# Optional: Fish API (D1 hybrid 才需要)
FISH_API_KEY=...
```

---

## 十、規則 / Boundary

1. **絕對不動 mirror-app code** ── 想看用 `/codex-audit-mirror-app`, 不要 Edit/Write 進 mirror-app 資料夾
2. **不在 web 跑 PPG-DTW 評估** ── client 只上傳錄音, 評估你本機後 batch
3. **不在 web 接 ElevenLabs IVC / Fish IVC** ── 都在你本機 / mirror-app 處理, web 只 fetch 預生成的 audio
4. **受試者隱私**: research_recordings bucket 永遠 private; signed URL 短 TTL
5. **不洩 service_role_key** ── 永遠在 server-only context (Route Handler 或 Server Component) 使用
