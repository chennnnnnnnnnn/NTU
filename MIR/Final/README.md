# Multi-Emotion Singing Voice Synthesis via EMelodyGen and DiffSinger

**論文投稿：ISMIR 2025**

> 陳博文・王家欣・張允妍  
> 國立臺灣大學 通訊工程學系

---

## 專案簡介

本專案提出一套端對端的多情緒歌聲合成（SVS）pipeline，整合了情緒條件化旋律生成系統 **EMelodyGen** 與兩階段神經網路歌聲合成框架 **DiffSinger**。

核心挑戰在於：如何在符號生成（Symbolic Generation）與聲學合成（Acoustic Synthesis）兩個階段之間，保持一致且可感知的情緒表達。本系統以下列三項機制解決此問題：

1. 以大型語言模型 **Qwen** 對訓練資料進行零樣本情緒標註（Zero-shot Annotation）
2. 建立確定性樂譜轉換模組（Deterministic Score Conversion），確保歌詞與音符對齊的穩定性
3. 採用兩階段聲學建模：Non-diffusion 粗糙預測 + Shallow Diffusion 細節精修

---

## 系統架構

```
Happy / Sad (Emotion Tag)
        │
        ▼
EMelodyGen 輸出 (ABC Notation)
        │
        ▼
表現力後處理 (Tempo / Pitch / Volume 映射)
        │
        ▼
歌詞對齊（字數統計 / Lyric Assignment）
        │
        ▼
樂譜格式轉換 (ABC → MusicXML / DiffSinger DS format)
        │
        ▼
DiffSinger 推理 / 合成
        │
        ▼
情緒化歌聲音訊輸出
```

### 情緒特徵模板（EMelodyGen）

| 情緒  | Q   | 調性  | PitchSD | Tempo        | Octave    | Volume       |
|-------|-----|-------|---------|--------------|-----------|--------------|
| Happy | Q1  | Major | High    | 100–120 BPM* | No Change | +5 dB        |
| Sad   | Q4  | Major | L       | 40–69 BPM    | No Change | No Change    |

> \* 原始 EMelodyGen 設定為 160–184 BPM，本實驗調降至 100–120 BPM 以提升人聲可唱性。

---

## 訓練策略（DiffSinger 變體）

本實驗訓練三個版本，以不同學習率配置與主模型凍結策略進行比較：

| 版本 | base_lr              | mult         | emo_lr               | 主模型狀態         |
|------|----------------------|--------------|----------------------|--------------------|
| V2   | 0.0001               | 50×          | 0.005                | Frozen             |
| V3   | 0.00005              | 10×          | 0.0005               | Unfrozen           |
| V4   | 0.0001 → 0.00003     | 50× → 1×     | 0.005 → 0.00003      | Frozen → Unfrozen  |

### 訓練結果摘要

| 指標                   | V2     | V3     | V4     |
|------------------------|--------|--------|--------|
| Mel Loss               | 0.0339 | 0.0333 | 0.0334 |
| Total Loss             | 0.0340 | 0.0458 | 0.0459 |
| Best Convergence Steps | 1923   | 6367   | 443    |

---

## 模型配置

- **模型大小**：`hidden_size=256`，`enc_layers=4`，`dec_layers=4`，`num_heads=4`
- **FFN kernel**：`enc_ffn_kernel_size=9`，`dec_ffn_kernel_size=9`
- **聲學特徵**：`mel_bins=80`
- **Diffusion**：`timesteps=1000`
- **音訊**：取樣率 24 kHz，`hop_size=128`
- **批次大小**：48（動態）

### 損失函數

```
L_total = L_mel + L_dur + L_f0 + L_diff + L_emo
```

- `L_mel` = MSE(mel_coarse, mel_gt)
- `L_dur` = MSE(dur_pred, dur_gt)
- `L_f0`  = MSE(f0_pred, f0_gt)
- `L_diff` = MSE(noise_pred, noise)
- `L_emo` = CrossEntropyLoss(prob, emotion_id) + margin-based contrastive term

---

## 資料集

- **DiffSinger 訓練集**：OpenCpop 資料集
- **情緒標註**：使用 `Qwen/Qwen2-Audio-7B-Instruct` 進行零樣本分類，輸出 happy (0) / sad (1)
- **標籤分佈**：Happy 2,998 筆（79.9%）/ Sad 756 筆（20.1%）
- **資料切割**：Train / Validation / Test = 90% / 5% / 5%

---

## 音訊樣本

### 根目錄

| 檔案 | 說明 |
|------|------|
| `Q1_Original.wav` | Q1（Happy 情緒）原始歌聲參考音訊 |

### `20251218_010122/`（批次推理對比）

每首歌曲包含以下變體（Q1 / Q4 各一組）：

| 命名格式 | 說明 |
|----------|------|
| `Q*_original.wav` | 原始參考音訊 |
| `Q*_V2_baseline.wav` | V2 訓練版本，無情緒條件 |
| `Q*_V2_happy.wav` | V2 訓練版本，Happy 情緒 |
| `Q*_V2_sad.wav` | V2 訓練版本，Sad 情緒 |
| `Q*_V3_*.wav` | V3 訓練版本（同上） |
| `Q*_V4_*.wav` | V4 訓練版本（同上） |

### `batch_comparison/`

包含多個時間戳子目錄（`20251218_002241` 等），儲存不同批次實驗的對比推理結果。

---

## 推理時手動縮放（Manual Scaling）

為放大可感知的情緒差異，推理時可額外套用以下縮放配置：

| 配置 | emotion_scale | f0_semitone_shift | f0_sentence_shift | duration_scale | 說明 |
|------|:---:|:---:|:---:|:---:|------|
| original | 0.0 | – | – | – | 無情緒嵌入 |
| baseline | 1.0 | – | – | – | – |
| happy | 5.0 | +1.5 | +1.5 | 0.9 | 音高上移 + 加快 |
| sad | 5.0 | -1.5 | -1.5 | 1.15 | 音高下移 + 放慢 |

---

## 客觀評估指標

對每個 WAV 檔分析前 120 秒，提取以下描述子：

- `spec_centroid_hz_mean`：頻譜重心（反映音色亮度）
- `rolloff85_hz_mean`：85% rolloff 頻率（反映響度）
- `rms_dbfs`：RMS 響度（dBFS）
- `silence_ratio_<-60dB`：靜音比例

---

## 主要發現

1. **EMelodyGen 生成旋律的可唱性有限**：生成的旋律結構較為複雜，原始速度（160–184 BPM）下歌詞清晰度不佳，降速至 100–120 BPM 後可唱性明顯提升。
2. **V3 的嵌入分離度最高**（Happy vs. Sad L2 距離 = 0.86），但主觀聆聽的情緒對比未必最明顯。
3. **推理時 Manual Scaling** 是在不重新訓練的情況下有效放大情緒感知差異的實用手段。

---

## 成員貢獻

| 成員 | 貢獻 |
|------|------|
| **王家欣** | EMelodyGen 生成 pipeline 與取樣配置；表現力後處理（tempo/pitch/volume 映射）及樂譜格式轉換 |
| **陳博文** | DiffSinger 推理與 vocoder 整合；客觀與主觀評估、結果表格與圖表製作 |
| **張允妍** | DiffSinger pipeline 實作；歌詞對齊規則優化 |

---

## 參考文獻

1. M. Zhou, X. Li, F. Yu, and W. Li, "EMelodyGen: Emotion-conditioned melody generation in ABC notation with the musical feature template," arXiv.
2. J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, "DiffSinger: Singing voice synthesis via shallow diffusion mechanism," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 10, 2022, pp. 11020–11028.
