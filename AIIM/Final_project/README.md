# TCGA 乳癌存活預測分析專案

本專案使用 TCGA (The Cancer Genome Atlas) 乳癌資料集，比較不同特徵組合對存活預測的效能。

## 專案結構

```
AIIM/
├── data_clinical_patient.txt              # 臨床資料
├── data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt  # 基因表達資料
├── README.md                              # 本文件
├── requirements.txt                       # Python 套件需求
│
├── Clinical/                              # 臨床特徵分析 (舊版，無右設限)
├── Clinical2/                             # 臨床特徵分析 (新版，有右設限)
├── Gene/                                  # 基因特徵分析 (舊版，無右設限)
├── Gene2/                                 # 基因特徵分析 (新版，有右設限)
├── Clinical_Gene/                         # 臨床+基因分析 (舊版，無右設限)
├── Clinical_Gene_2/                       # 臨床+基因分析 (新版，有右設限)
└── Summary/                               # 綜合分析結果
```

## 資料說明

### 原始資料
- **臨床資料**: 1097 位病患的臨床資訊
- **基因資料**: 20531 個基因的 z-score 表達值

### 前處理步驟
1. 刪除男性病患
2. 刪除年齡缺失
3. 刪除 Stage 問題 (Stage X, [Discrepancy], [Not Available])
4. Stage 分組: I→1, II→2, III→3, IV→4

### 基因集定義
| 基因集 | 基因數 | 說明 |
|--------|--------|------|
| PAM50 | 47/50 | 缺少 FOXC1, KRT17, MDM2 |
| Oncotype21 | 21/21 | 完整 |
| Union | 57 | PAM50 ∪ Oncotype21 |
| Intersection | 11 | PAM50 ∩ Oncotype21 |

**Intersection 基因**: BAG1, BCL2, BIRC5, CCNB1, ERBB2, ESR1, GRB7, MKI67, MMP11, MYBL2, PGR

## 分析方法

### 預測任務
- **DFS (Disease-Free Survival)**: 無病存活期預測
- **OS (Overall Survival)**: 總生存期預測

### 二值化分類
對 1-10 年進行二值化:
- 正例 (label=1): 存活時間 >= N 年
- 負例 (label=0): 存活時間 < N 年 且 發生事件

### 右設限處理 (新版)
刪除存活時間 < N 年且狀態為「存活/無病」的樣本，避免資訊洩漏。

### 機器學習模型
- Logistic Regression
- Ridge Classifier
- K-Nearest Neighbors
- Support Vector Machine (RBF kernel)
- Random Forest
- Gradient Boosting
- Multi-Layer Perceptron

### 交叉驗證
5-fold Stratified K-Fold Cross-Validation

## 使用方式

### 環境設置
```bash
pip install -r requirements.txt
```

### 執行分析

#### Clinical2 (臨床特徵，有右設限)
```bash
cd Clinical2
python clinical_classification_with_censoring.py
```

#### Gene2 (基因特徵，有右設限)
```bash
cd Gene2
python gene_preprocessing_with_censoring.py
```

#### Clinical_Gene_2 (臨床+基因，有右設限)
```bash
cd Clinical_Gene_2
python clinical_gene_classification_with_censoring.py
```

#### 綜合分析
```bash
cd Summary
python analyze_all_results.py
```

## 結果摘要

### 各版本比較

| 資料夾 | 版本 | 右設限 | DFS Mean AUC | OS Mean AUC |
|--------|------|--------|--------------|-------------|
| Clinical | 舊版 | 無 | 0.534 | 0.555 |
| Clinical2 | 新版 | **有** | **0.584** | 0.653 |
| Gene | 舊版 | 無 | 0.568 | 0.582 |
| Gene2 | 新版 | **有** | 0.547 | 0.618 |
| Clinical_Gene | 舊版 | 無 | 0.578 | 0.599 |
| Clinical_Gene_2 | 新版 | **有** | 0.572 | **0.671** |

### 最佳配置

| 預測任務 | 建議方案 | Mean AUC | Max AUC |
|----------|----------|----------|---------|
| DFS | Clinical2 | 0.584 | 0.726 |
| OS | Clinical_Gene_2 | 0.671 | 0.891 |

### 新版改善幅度

| 比較 | DFS | OS |
|------|-----|-----|
| Clinical → Clinical2 | +9.5% | +17.6% |
| Gene → Gene2 | -3.7% | +6.1% |
| Clinical_Gene → Clinical_Gene_2 | -1.1% | +12.1% |

## 主要結論

1. **使用新版 (有右設限處理)**: 結果更準確可靠
2. **DFS 預測**: 僅需臨床特徵 (Age + Stage)，Clinical2 效果最佳
3. **OS 預測**: 結合臨床+基因特徵，Clinical_Gene_2 效果最佳
4. **短期預測更準確**: 1-3 年預測效果最好

## 輸出檔案說明

### 資料檔
- `*_with_status.csv`: 含 Status 欄位的資料
- `*_with_pca.csv`: 含 PCA 特徵的資料
- `*_clinical_gene_pca.csv`: 臨床+基因+PCA 資料

### 結果檔
- `*_results_censored.csv`: 分類結果 (有右設限)
- `*_classification_results.csv`: 分類結果 (無右設限)
- `*_sample_counts.csv`: 樣本數統計

### 圖表
- `*_auc_comparison.png`: AUC 比較圖
- `*_auc_heatmap.png`: AUC 熱力圖
- `*_sample_counts.png`: 樣本數分布圖

## 參考資料

- TCGA Breast Cancer Dataset
- PAM50 Gene Signature
- Oncotype DX 21-Gene Assay
