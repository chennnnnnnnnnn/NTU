# Clinical_Gene 資料前處理完整流程說明

## 📋 總覽

這個前處理流程整合了**基因表達資料**和**臨床資料**，生成 8 個訓練用 CSV 檔案（4 個基因集 × 2 個目標變數）。

**主要目標**: 結合臨床特徵（年齡、分期）與基因表達資料來預測乳癌病人的存活情況（DFS 和 OS）

---

## 🔄 完整處理流程

### **步驟 1: 基因資料讀取**

**輸入**: `data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt`

```
原始形狀: (20,531 基因, 1,102 樣本)
```

**內容**:
- 20,531 個基因的 z-score 標準化表達量
- 1,102 個 TCGA 乳癌樣本

---

### **步驟 2: 轉置並清理**

#### 2.1 轉置資料
```
轉置前: (20,531 基因, 1,102 樣本)
轉置後: (1,101 樣本, 20,531 基因)  # 樣本變成行，基因變成列
```

#### 2.2 刪除 Entrez_Gene_Id 行
```
(1,101 樣本) → (1,100 樣本)
```
原因: 這行只包含基因 ID，不是樣本資料

#### 2.3 設定欄位名
```
使用第一行（Hugo_Symbol）作為欄位名
刪除第一行後: (1,100 樣本, 20,532 欄位)
```

#### 2.4 刪除全 NaN 的基因欄位
```
刪除 319 個全 NaN 的基因欄位
結果: (1,100 樣本, 20,213 欄位)
保留基因數: 20,212 個 (扣除 Hugo_Symbol 欄位)
```

**原因**: 這些基因在所有樣本中都沒有表達量資料，無法使用

---

### **步驟 3: 處理重複樣本 (-01 vs -06)**

**TCGA 樣本命名規則**:
- `-01`: 原發腫瘤 (Primary Tumor)
- `-06`: 轉移腫瘤 (Metastatic)

**處理邏輯**:
```
-01 樣本數: 1,093
-06 樣本數: 7
同時有 -01 和 -06 的病人: 7 人
```

**刪除策略**: 如果同一病人同時有 -01 和 -06 樣本，只保留 -01（原發腫瘤）

**被刪除的 7 個樣本**:
```
TCGA-AC-A6IX-06 → 保留 TCGA-AC-A6IX-01
TCGA-BH-A18V-06 → 保留 TCGA-BH-A18V-01
TCGA-BH-A1ES-06 → 保留 TCGA-BH-A1ES-01
TCGA-BH-A1FE-06 → 保留 TCGA-BH-A1FE-01
TCGA-E2-A15A-06 → 保留 TCGA-E2-A15A-01
TCGA-E2-A15E-06 → 保留 TCGA-E2-A15E-01
TCGA-E2-A15K-06 → 保留 TCGA-E2-A15K-01
```

**結果**:
```
刪除後: (1,093 樣本, 20,213 欄位)
唯一病人數: 1,093 人
基因數: 20,212 個
```

---

### **步驟 4: 臨床資料處理**

**輸入**: `data_clinical_patient.txt`

#### 4.1 讀取並選擇欄位
```
原始: (1,097 人, 111 欄位)
選擇 6 個欄位後: (1,097 人)
```

**保留的欄位**:
1. Patient Identifier
2. Sex
3. Diagnosis Age
4. Neoplasm Disease Stage American Joint Committee on Cancer Code
5. Disease Free (Months)
6. Overall Survival (Months)

#### 4.2 刪除 1 - 男性

**原因**: 本研究專注於女性乳癌

```
性別分布:
  Female: 1,085 人
  Male: 12 人

刪除男性後: 1,085 人
```

#### 4.3 刪除 2 - 年齡缺失

```
年齡統計:
  有效值: 1,096 人
  缺失值: 1 人 (TCGA-OL-A66H)
  範圍: 26-90 歲
  平均: 58.4 歲

刪除年齡缺失後: 1,084 人
```

#### 4.4 刪除 3 - 分期問題

**問題類型**:
- 不是 "Stage" 開頭: 11 人
- Stage X (無法確定分期): 13 人

**有問題的分期**:
```
Stage X: 13 人
[Discrepancy]: 6 人
[Not Available]: 5 人
總計: 24 人
```

```
刪除分期問題後: 1,060 人
```

#### 4.5 建立 Stage_Group (1-4)

**映射規則** (重要：必須按長度從長到短檢查！):
```python
if stage.startswith("STAGE IV"):   → Stage_Group = 4
elif stage.startswith("STAGE III"): → Stage_Group = 3
elif stage.startswith("STAGE II"):  → Stage_Group = 2
elif stage.startswith("STAGE I"):   → Stage_Group = 1
```

**為什麼順序重要？**
- 如果先檢查 "STAGE II"，"STAGE IIIA" 會被錯誤匹配為 Stage 2
- 必須從最長的字串開始檢查

**Stage_Group 分布**:
```
Stage 1: 182 人
  ├─ Stage I: 90
  ├─ Stage IA: 86
  └─ Stage IB: 6

Stage 2: 613 人
  ├─ Stage II: 5
  ├─ Stage IIA: 354
  └─ Stage IIB: 254

Stage 3: 246 人  ✓ (修正後正確顯示)
  ├─ Stage III: 2
  ├─ Stage IIIA: 154
  ├─ Stage IIIB: 26
  └─ Stage IIIC: 64

Stage 4: 19 人
  └─ Stage IV: 19
```

#### 4.6 處理生存時間 0 值

**問題**: 生存時間為 0 在統計上無法使用（log 轉換會出錯）

```
DFS (Disease Free Survival):
  0 值: 12 人
  負值: 1 人
  缺失: 87 人

OS (Overall Survival):
  0 值: 12 人
  負值: 1 人
  缺失: 1 人

處理方式: 將 0 值替換為 0.01
```

**臨床資料處理完成**: 1,060 人

---

### **步驟 5: 合併基因與臨床資料**

**合併前統計**:
```
基因資料病人數: 1,093
臨床資料病人數: 1,060
交集 (兩者都有): 1,056 人
只有基因資料: 37 人 (沒有臨床資料)
只有臨床資料: 4 人 (沒有基因資料)
```

**Inner Join 合併**:
```
df_merged = pd.merge(臨床, 基因, on="Patient Identifier", how="inner")

結果: (1,056 人, 20,251 欄位)
  = 5 臨床欄位 + 20,246 基因欄位
```

**欄位組成**:
- Patient Identifier (1)
- Diagnosis Age (1)
- Stage_Group (1)
- Disease Free (Months) (1)
- Overall Survival (Months) (1)
- 基因表達量 (20,246)

---

### **步驟 6: 定義基因集**

#### PAM50 基因集 (50 個)
用於乳癌亞型分類的 50 個基因

**可用性**: 47/50 個
**缺少**: FOXC1, KRT17, MDM2

#### Oncotype DX 基因集 (21 個)
商業化的乳癌復發風險評估基因組

**可用性**: 19/21 個
**缺少**: CTSL2, GUS

#### Union (聯集) - 60 個
PAM50 ∪ Oncotype DX = 60 個不重複基因

**可用性**: 55/60 個

#### Intersection (交集) - 11 個
PAM50 ∩ Oncotype DX = 11 個共同基因

**可用性**: 11/11 個 ✓ (全部可用)

**交集基因清單**:
```
BAG1, BCL2, BIRC5, CCNB1, ERBB2, ESR1, GRB7,
MKI67, MMP11, MYBL2, PGR
```

---

### **步驟 7: PCA 和 CSV 生成**

#### 對每個基因集執行：

**7.1 分割基因**
```
選擇的基因: 該基因集的基因
其他基因: 剩餘的 20,246 - n 個基因 (用於 PCA)
```

**7.2 執行 PCA**
```
對「其他基因」進行主成分分析
提取前 200 個主成分 (PC1-PC200)
解釋變異量: ~78.4%
```

**為什麼對「其他基因」做 PCA？**
- 選擇的基因 (如 PAM50) 已經有明確的生物學意義，直接作為特徵
- 其他 20,000+ 個基因資訊量太大，用 PCA 降維為 200 維
- 這樣既保留了專家知識（選擇的基因），也利用了全基因組資訊（PCA）

**7.3 合併特徵**
```
最終特徵 = Patient ID + Target + Age + Stage_Group + 選擇的基因 + PC1-200
```

**7.4 分別生成 DFS 和 OS**

**DFS (Disease Free Survival)**:
```
刪除 DFS 無效的樣本 (NaN 或 < 0)
最終: 968 人
```

**OS (Overall Survival)**:
```
刪除 OS 無效的樣本 (NaN 或 < 0)
最終: 1,054 人
```

---

## 📊 最終輸出結果

### 生成的 8 個 CSV 檔案

| 資料夾 | 基因集 | DFS 樣本 | OS 樣本 | 總欄位數 |
|--------|--------|----------|---------|----------|
| folder1_pam50 | PAM50 (47) | 968 | 1,054 | 251 |
| folder2_oncotype21 | Oncotype DX (19) | 968 | 1,054 | 223 |
| folder3_union_pam50_oncotype | Union (55) | 968 | 1,054 | 259 |
| folder4_intersection_pam50_oncotype | Intersection (11) | 968 | 1,054 | 215 |

### 欄位結構 (以 folder1_pam50/dfs_merged_with_clinical_pca.csv 為例)

**總欄位數: 251**

| 類型 | 數量 | 欄位名稱 |
|------|------|----------|
| Patient ID | 1 | Patient Identifier |
| Target | 1 | Disease Free (Months) |
| 臨床特徵 | 2 | Diagnosis Age, Stage_Group |
| 選擇的基因 | 47 | ACTR3B, ANLN, BAG1, ... |
| PCA 成分 | 200 | PC1, PC2, PC3, ..., PC200 |

### Stage_Group 分布

**DFS (968 人)**:
- Stage 1: 167 人 (17.3%)
- Stage 2: 567 人 (58.6%)
- Stage 3: 224 人 (23.1%)
- Stage 4: 10 人 (1.0%)

**OS (1,054 人)**:
- Stage 1: 180 人 (17.1%)
- Stage 2: 610 人 (57.9%)
- Stage 3: 245 人 (23.2%)
- Stage 4: 19 人 (1.8%)

---

## ✅ 資料品質驗證

### 缺失值檢查
```
所有 8 個 CSV: 0 個缺失值 ✓
```

### 數據流追蹤

```
基因資料:
1,102 樣本 (原始)
→ 1,100 樣本 (刪除 Entrez_Gene_Id + 轉置)
→ 1,093 樣本 (刪除重複 -06)
→ 1,093 人 (最終)

臨床資料:
1,097 人 (原始)
→ 1,085 人 (刪除 12 個男性)
→ 1,084 人 (刪除 1 個年齡缺失)
→ 1,060 人 (刪除 24 個分期問題)

合併:
1,056 人 (Inner Join)

DFS 最終: 968 人 (刪除 88 個 DFS 無效)
OS 最終: 1,054 人 (刪除 2 個 OS 無效)
```

---

## 🔑 關鍵設計決策

### 1. **為什麼刪除男性？**
- 乳癌主要發生在女性，男性乳癌的生物學特性不同
- 只有 12 個男性樣本，數量太少無法單獨分析

### 2. **為什麼刪除 -06 樣本？**
- `-01` (原發腫瘤) 和 `-06` (轉移腫瘤) 是同一病人的不同樣本
- 保留原發腫瘤樣本更能代表疾病初期狀態

### 3. **為什麼將 Stage 合併為 1-4？**
- Stage IA/IB/I 都屬於早期，合併為 Stage 1
- 減少類別數，提高模型穩定性
- 符合臨床實務中的大分類

### 4. **為什麼對「其他基因」做 PCA？**
- 選擇的基因（如 PAM50）有明確生物學意義，直接使用
- 剩餘 20,000+ 基因資訊量太大，用 PCA 降維
- 既保留專家知識，也利用全基因組資訊

### 5. **為什麼 DFS 和 OS 樣本數不同？**
- DFS 有更多缺失值 (87 個)，OS 只有 1 個缺失
- 臨床上更容易追蹤總生存時間 (OS)，疾病復發時間 (DFS) 較難確定

---

## 📝 與 Gene-only 流程的差異

| 項目 | Gene-only | Clinical_Gene |
|------|-----------|---------------|
| 臨床特徵 | 無 | Age + Stage_Group |
| 刪除條件 | 只刪男性 | 男性 + 年齡缺失 + 分期問題 |
| DFS 樣本數 | 989 人 | 968 人 (-21) |
| OS 樣本數 | 1,080 人 | 1,054 人 (-26) |
| 特徵數 | 基因 + PC | Age + Stage + 基因 + PC |

---

## 🎯 下一步：模型訓練

使用這 8 個 CSV 檔案，可以進行：

1. **二元分類模型訓練** (1-10 年生存預測)
2. **比較不同基因集的預測效果**
3. **評估臨床特徵的重要性**
4. **交叉驗證和超參數優化**

主要模型:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Neural Network
- XGBoost
- LightGBM

評估指標:
- ROC AUC
- F1 Score
- Precision / Recall
- Youden Index

---

**處理日期**: 2025-12-12
**腳本**: `clinical_gene_preprocessing.py`
**驗證**: `verify_preprocessing_details.py`
