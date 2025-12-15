"""
正確順序的基因資料預處理與二值化分類訓練
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

print("=" * 80)
print("開始執行正確順序的資料預處理與二值化分類訓練")
print("=" * 80)

# ============================================================================
# 步驟 1-3: 載入與預處理基因資料
# ============================================================================
print("\n[步驟 1-3] 載入與預處理基因資料...")
df_gene = pd.read_csv("data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt", sep="\t")
print(f"原始資料形狀: {df_gene.shape}")

# 轉置與處理
df_gene = df_gene.T.drop(index='Entrez_Gene_Id')
df_gene = df_gene.reset_index()
df_gene.columns = df_gene.iloc[0]
df_gene = df_gene.iloc[1:].reset_index(drop=True)

# 刪除含 NaN 的欄位
nan_count = df_gene.isna().sum()
df_gene = df_gene.loc[:, nan_count == 0].copy()
print(f"刪除含 NaN 欄位後: {df_gene.shape}")

# 刪除重複樣本
hs = df_gene["Hugo_Symbol"].astype(str)
patient_id = hs.str[:12]
is_01 = hs.str.endswith("-01")
is_06 = hs.str.endswith("-06")
pids_with_01 = set(patient_id[is_01])
drop_mask = is_06 & patient_id.isin(pids_with_01)
df_gene = df_gene.loc[~drop_mask].copy()
print(f"刪除重複樣本後: {df_gene.shape}")

# ============================================================================
# 步驟 4: 載入並處理臨床資料
# ============================================================================
print("\n[步驟 4] 載入臨床資料...")
df_clinical = pd.read_csv("data_clinical_patient.txt", sep="\t")
df_clinical = df_clinical[['Patient Identifier', 'Sex', 'Disease Free (Months)',
                           'Overall Survival (Months)']][4:]

# 轉換為數值並將 0 改為 0.01
for col in ['Disease Free (Months)', 'Overall Survival (Months)']:
    df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')
    df_clinical.loc[df_clinical[col] == 0, col] = 0.01

# 刪除男性
sex_series = df_clinical["Sex"].astype(str).str.strip().str.lower()
male_mask = sex_series.isin(["male", "m"])
df_clinical = df_clinical.loc[~male_mask].copy()
print(f"刪除男性後臨床資料: {df_clinical.shape}")

# ============================================================================
# 步驟 5: 合併基因與臨床資料
# ============================================================================
print("\n[步驟 5] 合併基因與臨床資料...")
df_gene["Patient Identifier"] = df_gene["Hugo_Symbol"].astype(str).str[:12]
gene_cols = [c for c in df_gene.columns if c not in ["Hugo_Symbol", "Patient Identifier"]]
df_gene_clean = df_gene[["Patient Identifier"] + gene_cols].copy()

# 準備 DFS 和 OS 資料集
df_dfs_clinical = df_clinical[['Patient Identifier', 'Disease Free (Months)']].dropna()
df_os_clinical = df_clinical[['Patient Identifier', 'Overall Survival (Months)']].dropna()

df_dfs_full = pd.merge(df_dfs_clinical, df_gene_clean, on="Patient Identifier", how="inner")
df_os_full = pd.merge(df_os_clinical, df_gene_clean, on="Patient Identifier", how="inner")

print(f"DFS 資料集: {df_dfs_full.shape}")
print(f"OS 資料集: {df_os_full.shape}")

# ============================================================================
# 步驟 6: 定義基因集
# ============================================================================
print("\n[步驟 6] 定義基因集...")

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

print(f"PAM50 基因數: {len(pam50_present)}/{len(PAM50_GENES)}")
print(f"Oncotype21 基因數: {len(oncotype_present)}/{len(ONCOTYPE21_GENES)}")
print(f"聯集基因數: {len(union_genes)}")
print(f"交集基因數: {len(intersection_genes)}")

GENE_SETS = {
    "folder1_pam50": pam50_present,
    "folder2_oncotype21": oncotype_present,
    "folder3_union_pam50_oncotype": union_genes,
    "folder4_intersection_pam50_oncotype": intersection_genes
}

# ============================================================================
# 步驟 7: 創建資料夾並處理 PCA
# ============================================================================
print("\n[步驟 7] 創建資料夾並進行 PCA...")

for folder in GENE_SETS.keys():
    os.makedirs(folder, exist_ok=True)

def process_and_save_with_pca(df, target_col, selected_genes, other_genes, folder_name, file_prefix):
    """對資料進行 PCA 並儲存"""
    # 準備特徵
    X_selected = df[selected_genes].apply(pd.to_numeric, errors='coerce')
    X_other = df[other_genes].apply(pd.to_numeric, errors='coerce')

    # 刪除含 NaN 的樣本
    valid_mask = X_selected.notna().all(axis=1) & X_other.notna().all(axis=1)
    X_selected_clean = X_selected.loc[valid_mask]
    X_other_clean = X_other.loc[valid_mask]

    # PCA (最多 200 個成分)
    n_components = min(200, X_other_clean.shape[1], X_other_clean.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_other_clean.values)

    # 創建 PCA DataFrame
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, index=X_selected_clean.index, columns=pc_cols)

    # 合併：Patient Identifier + target + selected genes + PCA
    df_result = pd.concat([
        df.loc[valid_mask, ["Patient Identifier", target_col]],
        X_selected_clean,
        df_pca
    ], axis=1)

    # 儲存
    output_path = f"{folder_name}/{file_prefix}_merged_with_pca.csv"
    df_result.to_csv(output_path, index=False)
    print(f"  已儲存: {output_path}, 形狀: {df_result.shape}")

    return df_result

for folder_name, selected_genes in GENE_SETS.items():
    print(f"\n處理 {folder_name}...")
    other_genes = [g for g in gene_cols if g not in selected_genes]

    # DFS
    process_and_save_with_pca(
        df_dfs_full, "Disease Free (Months)",
        selected_genes, other_genes,
        folder_name, "dfs"
    )

    # OS
    process_and_save_with_pca(
        df_os_full, "Overall Survival (Months)",
        selected_genes, other_genes,
        folder_name, "os"
    )

print("\n" + "=" * 80)
print("資料預處理完成！")
print("=" * 80)

# ============================================================================
# 步驟 8: 二值化分類訓練
# ============================================================================
print("\n[步驟 8] 開始二值化分類訓練...")

PC_COUNTS = 

YEARS = list(range(1, 11))  # 1~10 year

def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced",
        ),
        "RidgeClassifier": RidgeClassifier(class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=10),
        "SVC_rbf": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            solver="adam", max_iter=1000, random_state=42
        ),
    }

def compute_youden(y_true, y_pred):
    """計算 Youden index"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    J = sens + spec - 1
    return J, sens, spec

def run_classification_grid(df, target_col, task_name):
    """對不同年份 cutoff 進行二值化分類"""
    meta_cols = ["Patient Identifier", target_col]
    pc_cols_all = sorted([c for c in df.columns if c.startswith("PC")],
                         key=lambda x: int(x.replace("PC", "")))
    gene_cols = [c for c in df.columns if c not in meta_cols and c not in pc_cols_all]

    print(f"\n{'='*60}")
    print(f"任務: {task_name}")
    print(f"基因數: {len(gene_cols)}, PC數: {len(pc_cols_all)}")
    print(f"{'='*60}")

    months = pd.to_numeric(df[target_col], errors="coerce")
    df = df.copy()
    df[target_col] = months

    # 轉換特徵為數值
    for c in gene_cols + pc_cols_all:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    results = []
    classifiers = get_classifiers()

    for year in YEARS:
        cutoff_months = year * 12
        print(f"\n---- 年份: >= {year} 年 (cutoff = {cutoff_months} months) ----")

        # 建立二值標籤
        y_label = (months >= cutoff_months).astype("Int64")
        valid_mask_label = y_label.notna()
        df_year = df.loc[valid_mask_label].copy()
        y_label = y_label.loc[valid_mask_label]

        unique = y_label.unique()
        if len(unique) < 2:
            print(f"  ⚠ label 只有單一類別，跳過")
            continue

        for k in PC_COUNTS:
            k_use = min(k, len(pc_cols_all))
            cur_pc_cols = pc_cols_all[:k_use] if k_use > 0 else []
            feature_cols = gene_cols + cur_pc_cols

            X_df = df_year[feature_cols]
            y = y_label.copy()

            valid_mask = X_df.notna().all(axis=1) & y.notna()
            X = X_df.loc[valid_mask].astype(float).values
            y = y.loc[valid_mask].astype(int).values

            print(f"  K={k} (使用 {k_use} PCs), X shape={X.shape}")

            if X.shape[0] < 20 or len(np.unique(y)) < 2:
                print("    樣本太少或標籤單一，跳過")
                continue

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for model_name, model in classifiers.items():
                y_true_all = []
                y_pred_default_all = []
                y_score_all = []

                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model.fit(X_train, y_train)
                    y_pred_default = model.predict(X_test)
                    y_pred_default_all.append(y_pred_default)
                    y_true_all.append(y_test)

                    # 獲取預測分數
                    score = None
                    try:
                        score = model.predict_proba(X_test)[:, 1]
                    except:
                        try:
                            score = model.decision_function(X_test)
                        except:
                            pass

                    if score is not None:
                        y_score_all.append(score)

                y_true_all = np.concatenate(y_true_all)
                y_pred_default_all = np.concatenate(y_pred_default_all)

                # 計算默認指標
                acc_default = accuracy_score(y_true_all, y_pred_default_all)
                prec_default = precision_score(y_true_all, y_pred_default_all, zero_division=0)
                rec_default = recall_score(y_true_all, y_pred_default_all, zero_division=0)
                f1_default = f1_score(y_true_all, y_pred_default_all, zero_division=0)

                # AUC
                auc = np.nan
                if len(y_score_all) > 0:
                    y_score_all = np.concatenate(y_score_all)
                    try:
                        auc = roc_auc_score(y_true_all, y_score_all)
                    except:
                        pass
                else:
                    y_score_all = None

                # 掃描閾值找最佳 F1 和 Youden
                best_f1 = f1_default
                best_f1_th = np.nan
                best_f1_acc = acc_default
                best_f1_prec = prec_default
                best_f1_rec = rec_default

                best_J = np.nan
                best_J_th = np.nan
                best_J_sens = np.nan
                best_J_spec = np.nan

                if y_score_all is not None:
                    thresholds = np.linspace(0.05, 0.95, 19)
                    for th in thresholds:
                        y_pred_th = (y_score_all >= th).astype(int)

                        f1_th = f1_score(y_true_all, y_pred_th, zero_division=0)
                        if f1_th > best_f1:
                            best_f1 = f1_th
                            best_f1_th = th
                            best_f1_acc = accuracy_score(y_true_all, y_pred_th)
                            best_f1_prec = precision_score(y_true_all, y_pred_th, zero_division=0)
                            best_f1_rec = recall_score(y_true_all, y_pred_th, zero_division=0)

                        J, sens, spec = compute_youden(y_true_all, y_pred_th)
                        if np.isnan(best_J) or J > best_J:
                            best_J = J
                            best_J_th = th
                            best_J_sens = sens
                            best_J_spec = spec

                results.append({
                    "year": year,
                    "k_pc_requested": k,
                    "k_pc_used": k_use,
                    "n_features": X.shape[1],
                    "n_samples": X.shape[0],
                    "model": model_name,
                    "acc_default": acc_default,
                    "prec_default": prec_default,
                    "rec_default": rec_default,
                    "f1_default": f1_default,
                    "roc_auc": auc,
                    "best_f1": best_f1,
                    "best_f1_threshold": best_f1_th,
                    "acc_at_best_f1": best_f1_acc,
                    "prec_at_best_f1": best_f1_prec,
                    "rec_at_best_f1": best_f1_rec,
                    "best_youden": best_J,
                    "best_youden_threshold": best_J_th,
                    "sens_at_best_youden": best_J_sens,
                    "spec_at_best_youden": best_J_spec,
                })

    return pd.DataFrame(results)

# 對每個資料夾進行分類訓練
for folder_name in GENE_SETS.keys():
    print(f"\n{'='*80}")
    print(f"處理資料夾: {folder_name}")
    print(f"{'='*80}")

    # DFS
    dfs_path = f"{folder_name}/dfs_merged_with_pca.csv"
    if os.path.exists(dfs_path):
        df_dfs = pd.read_csv(dfs_path)
        dfs_results = run_classification_grid(
            df_dfs, "Disease Free (Months)", f"{folder_name} DFS"
        )
        dfs_results.to_csv(f"{folder_name}/dfs_classification_results.csv", index=False)
        print(f"✓ DFS 結果已儲存: {folder_name}/dfs_classification_results.csv")

    # OS
    os_path = f"{folder_name}/os_merged_with_pca.csv"
    if os.path.exists(os_path):
        df_os = pd.read_csv(os_path)
        os_results = run_classification_grid(
            df_os, "Overall Survival (Months)", f"{folder_name} OS"
        )
        os_results.to_csv(f"{folder_name}/os_classification_results.csv", index=False)
        print(f"✓ OS 結果已儲存: {folder_name}/os_classification_results.csv")

print("\n" + "=" * 80)
print("全部完成！")
print("=" * 80)
print("\n生成的檔案：")
for folder in GENE_SETS.keys():
    print(f"\n  {folder}/")
    print(f"    - dfs_merged_with_pca.csv")
    print(f"    - os_merged_with_pca.csv")
    print(f"    - dfs_classification_results.csv")
    print(f"    - os_classification_results.csv")
