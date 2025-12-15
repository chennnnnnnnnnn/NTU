"""
Clinical_Gene 資料集的二值化分類訓練（含右設限處理）
包含臨床特徵 (Age, Stage_Group) + 基因特徵 + PCA 特徵
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Clinical_Gene 二值化分類訓練（含右設限處理）")
print("=" * 80)

# ============================================================================
# 步驟 1: 載入與預處理基因資料
# ============================================================================
print("\n[步驟 1] 載入基因資料...")
df_gene = pd.read_csv("../data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt", sep="\t")
print(f"原始基因資料: {df_gene.shape}")

# 轉置與處理
df_gene = df_gene.T.drop(index='Entrez_Gene_Id')
df_gene = df_gene.reset_index()
df_gene.columns = df_gene.iloc[0]
df_gene = df_gene.iloc[1:].reset_index(drop=True)

# 刪除含 NaN 的欄位
nan_count = df_gene.isna().sum()
df_gene = df_gene.loc[:, nan_count == 0].copy()

# 刪除重複欄位名稱（保留第一個）
df_gene = df_gene.loc[:, ~df_gene.columns.duplicated()]

# 刪除重複樣本
hs = df_gene["Hugo_Symbol"].astype(str)
patient_id = hs.str[:12]
is_01 = hs.str.endswith("-01")
is_06 = hs.str.endswith("-06")
pids_with_01 = set(patient_id[is_01])
drop_mask = is_06 & patient_id.isin(pids_with_01)
df_gene = df_gene.loc[~drop_mask].copy()
df_gene["Patient Identifier"] = df_gene["Hugo_Symbol"].astype(str).str[:12]
print(f"處理後基因資料: {df_gene.shape}")

gene_cols = [c for c in df_gene.columns if c not in ["Hugo_Symbol", "Patient Identifier"]]

# ============================================================================
# 步驟 2: 載入臨床資料（含 Status 欄位）
# ============================================================================
print("\n[步驟 2] 載入臨床資料...")
df_clinical = pd.read_csv("../data_clinical_patient.txt", sep="\t")
df_clinical = df_clinical[['Patient Identifier', 'Sex', 'Diagnosis Age',
                           'Neoplasm Disease Stage American Joint Committee on Cancer Code',
                           'Disease Free (Months)', 'Disease Free Status',
                           'Overall Survival (Months)', 'Overall Survival Status']][4:]

# 欄位名稱
col_sex = "Sex"
col_age = "Diagnosis Age"
col_stage = "Neoplasm Disease Stage American Joint Committee on Cancer Code"
col_dfs = "Disease Free (Months)"
col_dfs_status = "Disease Free Status"
col_os = "Overall Survival (Months)"
col_os_status = "Overall Survival Status"

# 轉換數值
df_clinical[col_dfs] = pd.to_numeric(df_clinical[col_dfs], errors='coerce')
df_clinical[col_os] = pd.to_numeric(df_clinical[col_os], errors='coerce')
df_clinical[col_age] = pd.to_numeric(df_clinical[col_age], errors='coerce')

# 0 改成 0.01
df_clinical.loc[df_clinical[col_dfs] == 0, col_dfs] = 0.01
df_clinical.loc[df_clinical[col_os] == 0, col_os] = 0.01

# 識別問題資料
male_mask = df_clinical[col_sex].astype(str).str.strip().str.lower().isin(["male", "m"])
age_missing_mask = df_clinical[col_age].isna()
stage_series = df_clinical[col_stage].astype(str).str.strip()
stage_problem_mask = (~stage_series.str.startswith("Stage", na=False)) | stage_series.eq("Stage X")

# 刪除問題資料
problem_mask = male_mask | age_missing_mask | stage_problem_mask
df_clinical = df_clinical[~problem_mask].copy()
print(f"刪除問題資料後: {len(df_clinical)} 筆")

# Stage 分組
def stage_to_group(s):
    s = str(s).strip()
    if s in ["Stage I", "Stage IA", "Stage IB"]:
        return 1
    elif s in ["Stage II", "Stage IIA", "Stage IIB"]:
        return 2
    elif s in ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"]:
        return 3
    elif s == "Stage IV":
        return 4
    else:
        return None

df_clinical["Stage_Group"] = df_clinical[col_stage].apply(stage_to_group)
df_clinical["Diagnosis Age"] = df_clinical[col_age].astype(int)

print("Status 欄位值:")
print("  OS:", df_clinical[col_os_status].value_counts().to_dict())
print("  DFS:", df_clinical[col_dfs_status].value_counts().to_dict())

# ============================================================================
# 步驟 3: 定義基因集
# ============================================================================
print("\n[步驟 3] 定義基因集...")

PAM50_GENES = [
    "ACTR3B","ANLN","BAG1","BCL2","BIRC5","BLVRA","CCNB1","CCNE1","CDC20","CDC6",
    "CDH3","CENPF","CEP55","CXXC5","EGFR","ERBB2","ESR1","EXO1","FGFR4","FOXA1",
    "FOXC1","GPR160","GRB7","KIF2C","KRT14","KRT17","KRT5","MAPT","MDM2","MELK",
    "MIA","MKI67","MLPH","MMP11","MYBL2","MYC","NAT1","NDC80","NUF2","ORC6",
    "PGR","PHGDH","PTTG1","RRM2","SFRP1","SLC39A6","TMEM45B","TYMS","UBE2C","UBE2T"
]

ONCOTYPE21_GENES = [
    "GRB7", "ERBB2", "ESR1", "PGR", "BCL2", "SCUBE2",
    "MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2",
    "MMP11", "CTSV", "GSTM1", "CD68", "BAG1",
    "ACTB", "TFRC", "GAPDH", "GUSB", "RPLP0"
]

pam50_present = [g for g in PAM50_GENES if g in gene_cols]
oncotype_present = [g for g in ONCOTYPE21_GENES if g in gene_cols]
union_genes = pam50_present + [g for g in oncotype_present if g not in pam50_present]
intersection_genes = [g for g in pam50_present if g in ONCOTYPE21_GENES]

print(f"PAM50: {len(pam50_present)}/{len(PAM50_GENES)}")
print(f"Oncotype21: {len(oncotype_present)}/{len(ONCOTYPE21_GENES)}")
print(f"Union: {len(union_genes)}")
print(f"Intersection: {len(intersection_genes)}")

GENE_SETS = {
    "pam50": pam50_present,
    "oncotype21": oncotype_present,
    "union": union_genes,
    "intersection": intersection_genes
}

# ============================================================================
# 步驟 4: 合併資料並進行 PCA
# ============================================================================
print("\n[步驟 4] 合併資料並進行 PCA...")

def prepare_merged_data(df_clinical, df_gene, target_col, status_col, selected_genes, gene_cols):
    """合併臨床和基因資料，進行 PCA"""
    # 準備臨床資料
    clinical_cols = ['Patient Identifier', 'Diagnosis Age', 'Stage_Group', target_col, status_col]
    df_clin = df_clinical[clinical_cols].dropna(subset=[target_col]).copy()

    # 準備基因資料
    df_gene_clean = df_gene[["Patient Identifier"] + gene_cols].copy()

    # 合併
    df_merged = pd.merge(df_clin, df_gene_clean, on="Patient Identifier", how="inner")

    # 轉換基因欄位為數值
    for g in gene_cols:
        df_merged[g] = pd.to_numeric(df_merged[g], errors='coerce')

    # 刪除含 NaN 的樣本
    df_merged = df_merged.dropna().reset_index(drop=True)

    # PCA 處理非選定基因
    other_genes = [g for g in gene_cols if g not in selected_genes]
    X_other = df_merged[other_genes].values

    n_components = min(200, X_other.shape[1], X_other.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_other)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)

    # 組合最終資料
    df_result = pd.concat([
        df_merged[['Patient Identifier', 'Diagnosis Age', 'Stage_Group', target_col, status_col]],
        df_merged[selected_genes],
        df_pca
    ], axis=1)

    return df_result

# 準備各基因集資料
processed_data = {}
for gene_set_name, selected_genes in GENE_SETS.items():
    print(f"\n處理 {gene_set_name}...")

    # DFS
    df_dfs = prepare_merged_data(
        df_clinical, df_gene, col_dfs, col_dfs_status, selected_genes, gene_cols
    )
    df_dfs.to_csv(f"dfs_{gene_set_name}_clinical_gene_pca.csv", index=False)
    print(f"  DFS: {df_dfs.shape}")

    # OS
    df_os = prepare_merged_data(
        df_clinical, df_gene, col_os, col_os_status, selected_genes, gene_cols
    )
    df_os.to_csv(f"os_{gene_set_name}_clinical_gene_pca.csv", index=False)
    print(f"  OS: {df_os.shape}")

    processed_data[gene_set_name] = {
        'dfs': df_dfs,
        'os': df_os,
        'genes': selected_genes
    }

# ============================================================================
# 步驟 5: 分類訓練配置
# ============================================================================
print("\n" + "=" * 80)
print("[步驟 5] 開始分類訓練（含右設限處理）")
print("=" * 80)

PC_COUNTS = [0, 10, 20, 50, 100, 200]
YEARS = list(range(1, 11))

def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RidgeClassifier": RidgeClassifier(class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "SVC_rbf": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }

def compute_youden(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens + spec - 1, sens, spec

def run_classification_with_censoring(df, target_col, status_col, living_value,
                                       gene_cols_list, task_name):
    """含右設限處理的分類訓練"""
    clinical_cols = ["Diagnosis Age", "Stage_Group"]
    pc_cols_all = sorted([c for c in df.columns if c.startswith("PC")],
                         key=lambda x: int(x.replace("PC", "")))

    print(f"\n{'='*60}")
    print(f"任務: {task_name}")
    print(f"臨床特徵: {len(clinical_cols)}, 基因特徵: {len(gene_cols_list)}, PC數: {len(pc_cols_all)}")
    print(f"{'='*60}")

    months = pd.to_numeric(df[target_col], errors="coerce")
    df = df.copy()
    df[target_col] = months

    # 轉換特徵為數值
    for c in clinical_cols + gene_cols_list + pc_cols_all:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    results = []
    sample_counts = []
    classifiers = get_classifiers()

    for year in YEARS:
        cutoff_months = year * 12

        # 二值化標籤
        y_label = (months >= cutoff_months).astype(int)

        # 右設限處理：刪除 時間 < cutoff 且 Status = LIVING/DiseaseFree 的樣本
        censored_mask = (months < cutoff_months) & (df[status_col] == living_value)
        df_year = df[~censored_mask].copy()
        y_label = y_label[~censored_mask]

        n_total = len(df_year)
        n_pos = y_label.sum()
        n_neg = n_total - n_pos
        n_censored = censored_mask.sum()

        sample_counts.append({
            "Year": year,
            "Total": n_total,
            "Positive": n_pos,
            "Negative": n_neg,
            "Censored_removed": n_censored
        })

        print(f"\n---- Year >= {year} (樣本:{n_total}, 正:{n_pos}, 負:{n_neg}, 刪除右設限:{n_censored}) ----")

        if n_pos < 10 or n_neg < 10:
            print(f"  ⚠ 樣本太少，跳過")
            continue

        for k in PC_COUNTS:
            k_use = min(k, len(pc_cols_all))
            cur_pc_cols = pc_cols_all[:k_use] if k_use > 0 else []

            # 特徵：臨床 + 基因 + PCA
            feature_cols = clinical_cols + gene_cols_list + cur_pc_cols

            X_df = df_year[feature_cols]
            y = y_label.copy()

            valid_mask = X_df.notna().all(axis=1) & y.notna()
            X = X_df.loc[valid_mask].astype(float).values
            y = y.loc[valid_mask].astype(int).values

            if X.shape[0] < 20 or len(np.unique(y)) < 2:
                continue

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for model_name, model in classifiers.items():
                y_true_all, y_pred_all, y_score_all = [], [], []

                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_all.append(y_pred)
                    y_true_all.append(y_test)

                    try:
                        score = model.predict_proba(X_test)[:, 1]
                    except:
                        try:
                            score = model.decision_function(X_test)
                        except:
                            score = None

                    if score is not None:
                        y_score_all.append(score)

                y_true_all = np.concatenate(y_true_all)
                y_pred_all = np.concatenate(y_pred_all)

                acc = accuracy_score(y_true_all, y_pred_all)
                prec = precision_score(y_true_all, y_pred_all, zero_division=0)
                rec = recall_score(y_true_all, y_pred_all, zero_division=0)
                f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

                auc = np.nan
                if len(y_score_all) > 0:
                    y_score_all = np.concatenate(y_score_all)
                    try:
                        auc = roc_auc_score(y_true_all, y_score_all)
                    except:
                        pass

                # Threshold 掃描
                best_f1, best_f1_th = f1, np.nan
                best_J, best_J_th = np.nan, np.nan

                if isinstance(y_score_all, np.ndarray):
                    for th in np.linspace(0.05, 0.95, 19):
                        y_pred_th = (y_score_all >= th).astype(int)
                        f1_th = f1_score(y_true_all, y_pred_th, zero_division=0)
                        if f1_th > best_f1:
                            best_f1, best_f1_th = f1_th, th

                        J, _, _ = compute_youden(y_true_all, y_pred_th)
                        if np.isnan(best_J) or J > best_J:
                            best_J, best_J_th = J, th

                results.append({
                    "year": year,
                    "k_pc": k_use,
                    "n_clinical": len(clinical_cols),
                    "n_genes": len(gene_cols_list),
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                    "model": model_name,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": auc,
                    "best_f1": best_f1,
                    "best_f1_th": best_f1_th,
                    "best_youden": best_J,
                    "best_youden_th": best_J_th
                })

        # 印出 k=0 和 k=50 的最佳結果
        year_results = [r for r in results if r['year'] == year]
        if year_results:
            for k in [0, 50]:
                k_results = [r for r in year_results if r['k_pc'] == k]
                if k_results:
                    best = max(k_results, key=lambda x: x['roc_auc'] if not np.isnan(x['roc_auc']) else 0)
                    print(f"  K={k}: Best {best['model']} AUC={best['roc_auc']:.3f}")

    return pd.DataFrame(results), pd.DataFrame(sample_counts)

# ============================================================================
# 步驟 6: 執行分類訓練
# ============================================================================
all_results = {}

for gene_set_name, data in processed_data.items():
    print(f"\n{'='*80}")
    print(f"處理基因集: {gene_set_name}")
    print(f"{'='*80}")

    # DFS
    dfs_results, dfs_counts = run_classification_with_censoring(
        data['dfs'],
        col_dfs,
        col_dfs_status,
        "0:DiseaseFree",
        data['genes'],
        f"{gene_set_name} DFS (Clinical+Gene)"
    )
    dfs_results.to_csv(f"dfs_{gene_set_name}_results_censored.csv", index=False)
    dfs_counts.to_csv(f"dfs_{gene_set_name}_sample_counts.csv", index=False)

    # OS
    os_results, os_counts = run_classification_with_censoring(
        data['os'],
        col_os,
        col_os_status,
        "0:LIVING",
        data['genes'],
        f"{gene_set_name} OS (Clinical+Gene)"
    )
    os_results.to_csv(f"os_{gene_set_name}_results_censored.csv", index=False)
    os_counts.to_csv(f"os_{gene_set_name}_sample_counts.csv", index=False)

    all_results[gene_set_name] = {
        'dfs': dfs_results,
        'os': os_results
    }

print("\n" + "=" * 80)
print("全部完成！")
print("=" * 80)

# 列出輸出檔案
print("\n輸出檔案:")
import glob
for f in sorted(glob.glob("*.csv")):
    print(f"  {f}")
