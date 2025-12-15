"""
臨床+基因資料預處理
生成 8 個訓練用 CSV 檔案（4個基因集 × 2個目標變數）
包含: Patient Identifier, Target, Diagnosis Age, Stage_Group, Selected Genes, PC1-200
"""

import os
import pandas as pd
from sklearn.decomposition import PCA

print("=" * 80)
print("臨床+基因資料預處理")
print("=" * 80)

# ============================================================================
# 步驟 1-3: 基因資料處理（與原始流程相同）
# ============================================================================
print("\n[步驟 1-3: 基因資料處理]")

# 步驟 1: 讀取基因資料
print("步驟 1: 讀取基因資料...")
df_gene = pd.read_csv("../data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt", sep="\t")
print(f"  原始基因資料: {df_gene.shape}")

# 步驟 2: 轉置並清理
print("步驟 2: 轉置資料...")
df_gene = df_gene.T.drop(index='Entrez_Gene_Id')
df_gene = df_gene.reset_index()
df_gene.columns = df_gene.iloc[0]
df_gene = df_gene.iloc[1:].reset_index(drop=True)
print(f"  轉置後: {df_gene.shape}")

# 刪除全為 NaN 的基因欄位
nan_count = df_gene.isna().sum()
n_nan_cols = (nan_count == len(df_gene)).sum()
df_gene = df_gene.loc[:, nan_count == 0].copy()
print(f"  刪除 {n_nan_cols} 個全 NaN 的基因欄位")
print(f"  保留: {df_gene.shape}")

# 步驟 3: 處理重複樣本（-01 vs -06）
print("步驟 3: 處理重複樣本...")
hs = df_gene["Hugo_Symbol"].astype(str)
patient_id = hs.str[:12]
is_01 = hs.str.endswith("-01")
is_06 = hs.str.endswith("-06")
pids_with_01 = set(patient_id[is_01])
drop_mask = is_06 & patient_id.isin(pids_with_01)
n_dup = drop_mask.sum()
df_gene = df_gene.loc[~drop_mask].copy()
print(f"  刪除 {n_dup} 個重複的 -06 樣本")
print(f"  保留: {df_gene.shape}")

# 建立 Patient Identifier
df_gene["Patient Identifier"] = df_gene["Hugo_Symbol"].astype(str).str[:12]
gene_cols = [c for c in df_gene.columns if c not in ["Hugo_Symbol", "Patient Identifier"]]
df_gene_clean = df_gene[["Patient Identifier"] + gene_cols].copy()

print(f"\n基因資料清理完成: {df_gene_clean.shape}")
print(f"  病人數: {len(df_gene_clean)}")
print(f"  基因數: {len(gene_cols)}")

# ============================================================================
# 步驟 4: 臨床資料處理（新增刪除條件）
# ============================================================================
print("\n[步驟 4: 臨床資料處理（新增刪除條件）]")

# 讀取臨床資料
df_clinical = pd.read_csv("../data_clinical_patient.txt", sep="\t")
df_clinical = df_clinical[['Patient Identifier', 'Sex', 'Diagnosis Age',
                           'Neoplasm Disease Stage American Joint Committee on Cancer Code',
                           'Disease Free (Months)', 'Overall Survival (Months)']][4:]

print(f"原始臨床資料: {len(df_clinical)} 人")

# 記錄刪除情況
cleaning_log = []

# 刪除 1: 男性
col_sex = "Sex"
male_mask = df_clinical[col_sex].astype(str).str.strip().str.lower().isin(["male", "m"])
n_male = male_mask.sum()
cleaning_log.append({"step": "刪除男性", "n_deleted": n_male, "n_remaining": len(df_clinical) - n_male})
df_clinical = df_clinical.loc[~male_mask].copy()
print(f"刪除男性: {n_male} 人，剩餘 {len(df_clinical)} 人")

# 刪除 2: 年齡缺失
col_age = "Diagnosis Age"
age_numeric = pd.to_numeric(df_clinical[col_age], errors="coerce")
age_missing_mask = age_numeric.isna()
n_age_missing = age_missing_mask.sum()
cleaning_log.append({"step": "刪除年齡缺失", "n_deleted": n_age_missing, "n_remaining": len(df_clinical) - n_age_missing})
df_clinical = df_clinical.loc[~age_missing_mask].copy()
age_numeric = age_numeric.loc[~age_missing_mask]
print(f"刪除年齡缺失: {n_age_missing} 人，剩餘 {len(df_clinical)} 人")

# 刪除 3: 分期問題（不是 Stage 開頭或 Stage X）
col_stage = "Neoplasm Disease Stage American Joint Committee on Cancer Code"
stage_series = df_clinical[col_stage].astype(str).str.strip()
stage_problem_mask = (~stage_series.str.startswith("Stage", na=False)) | stage_series.eq("Stage X")
n_stage_problem = stage_problem_mask.sum()
cleaning_log.append({"step": "刪除分期問題", "n_deleted": n_stage_problem, "n_remaining": len(df_clinical) - n_stage_problem})
df_clinical = df_clinical.loc[~stage_problem_mask].copy()
stage_series = stage_series.loc[~stage_problem_mask]
print(f"刪除分期問題: {n_stage_problem} 人，剩餘 {len(df_clinical)} 人")

# 建立 Stage_Group (1-4)
print("\n建立 Stage_Group...")
df_clinical[col_age] = pd.to_numeric(df_clinical[col_age], errors="coerce").astype(int)

def map_stage_to_group(stage_str):
    """將 Stage 映射到 1-4 的群組"""
    stage_str = str(stage_str).strip().upper()
    # 必須按照長度從長到短檢查，避免 "STAGE III" 被誤判為 "STAGE II"
    if stage_str.startswith("STAGE IV"):
        return 4  # Stage IV, IVA, IVB
    elif stage_str.startswith("STAGE III"):
        return 3  # Stage III, IIIA, IIIB, IIIC
    elif stage_str.startswith("STAGE II"):
        return 2  # Stage II, IIA, IIB
    elif stage_str.startswith("STAGE I"):
        return 1  # Stage I, IA, IB
    else:
        return None

df_clinical["Stage_Group"] = df_clinical[col_stage].apply(map_stage_to_group)
print(f"Stage_Group 分布:\n{df_clinical['Stage_Group'].value_counts().sort_index()}")

# 處理生存時間的 0 值
print("\n處理生存時間的 0 值...")
col_os = "Overall Survival (Months)"
col_dfs = "Disease Free (Months)"

os_numeric = pd.to_numeric(df_clinical[col_os], errors="coerce")
dfs_numeric = pd.to_numeric(df_clinical[col_dfs], errors="coerce")

df_clinical[col_os] = os_numeric.copy()
df_clinical.loc[df_clinical[col_os] == 0, col_os] = 0.01

df_clinical[col_dfs] = dfs_numeric.copy()
df_clinical.loc[df_clinical[col_dfs] == 0, col_dfs] = 0.01

print(f"臨床資料處理完成: {len(df_clinical)} 人")

# 儲存清理日誌
cleaning_df = pd.DataFrame(cleaning_log)
cleaning_df.to_csv("cleaning_log.csv", index=False)
print(f"\n清理日誌已儲存至: cleaning_log.csv")

# ============================================================================
# 步驟 5: 合併基因與臨床資料
# ============================================================================
print("\n[步驟 5: 合併基因與臨床資料]")

# 只保留需要的臨床欄位
df_clinical_for_merge = df_clinical[["Patient Identifier", col_age, "Stage_Group", col_dfs, col_os]].copy()

# 合併
df_merged = pd.merge(df_clinical_for_merge, df_gene_clean, on="Patient Identifier", how="inner")
print(f"合併後: {df_merged.shape}")
print(f"  病人數: {len(df_merged)}")
print(f"  總欄位數: {df_merged.shape[1]}")

# ============================================================================
# 步驟 6: 定義基因集
# ============================================================================
print("\n[步驟 6: 定義基因集]")

# PAM50
pam50_genes = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20",
    "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1",
    "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5",
    "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1",
    "NDC80", "NUF2", "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6",
    "TMEM45B", "TYMS", "UBE2C", "UBE2T"
]

# Oncotype DX (21 genes)
oncotype_genes = [
    # HER2 / Estrogen modules
    "GRB7", "ERBB2",        # HER2 module (HER2 = ERBB2)
    "ESR1", "PGR", "BCL2", "SCUBE2",  # Estrogen module

    # Proliferation module
    "MKI67",          # Ki67
    "AURKA",          # STK15
    "BIRC5",
    "CCNB1",
    "MYBL2",

    # Invasion module
    "MMP11",
    "CTSV",           # Cathepsin V (原為 CTSL2，已修正)

    # Other cancer genes
    "GSTM1",
    "CD68",
    "BAG1",

    # Reference genes
    "ACTB",
    "TFRC",
    "GAPDH",
    "GUSB",           # Beta-Glucuronidase (原為 GUS，已修正)
    "RPLP0",
]

# Union
union_genes = list(set(pam50_genes) | set(oncotype_genes))
print(f"Union 基因數: {len(union_genes)}")

# Intersection
intersection_genes = list(set(pam50_genes) & set(oncotype_genes))
print(f"Intersection 基因數: {len(intersection_genes)}")

# 檢查基因是否存在
all_gene_cols = [c for c in df_merged.columns if c not in ["Patient Identifier", col_age, "Stage_Group", col_dfs, col_os]]

pam50_available = [g for g in pam50_genes if g in all_gene_cols]
oncotype_available = [g for g in oncotype_genes if g in all_gene_cols]
union_available = [g for g in union_genes if g in all_gene_cols]
intersection_available = [g for g in intersection_genes if g in all_gene_cols]

print(f"\nPAM50: {len(pam50_available)}/{len(pam50_genes)} 個基因可用")
print(f"Oncotype DX: {len(oncotype_available)}/{len(oncotype_genes)} 個基因可用")
print(f"Union: {len(union_available)}/{len(union_genes)} 個基因可用")
print(f"Intersection: {len(intersection_available)}/{len(intersection_genes)} 個基因可用")

gene_sets = {
    "pam50": pam50_available,
    "oncotype21": oncotype_available,
    "union_pam50_oncotype": union_available,
    "intersection_pam50_oncotype": intersection_available
}

# ============================================================================
# 步驟 7: 對每個基因集生成 CSV（包含 PCA）
# ============================================================================
print("\n[步驟 7: 生成訓練用 CSV 檔案]")

folder_names = [
    "folder1_pam50",
    "folder2_oncotype21",
    "folder3_union_pam50_oncotype",
    "folder4_intersection_pam50_oncotype"
]

gene_set_keys = ["pam50", "oncotype21", "union_pam50_oncotype", "intersection_pam50_oncotype"]

N_PC = 200  # PCA 成分數

for folder_name, gene_set_key in zip(folder_names, gene_set_keys):
    print(f"\n處理: {folder_name}")

    # 建立資料夾
    os.makedirs(folder_name, exist_ok=True)

    # 選擇的基因
    selected_genes = gene_sets[gene_set_key]

    # 未被選的基因（用於 PCA）
    other_genes = [g for g in all_gene_cols if g not in selected_genes]

    print(f"  選擇基因數: {len(selected_genes)}")
    print(f"  其他基因數: {len(other_genes)}")

    # 準備資料用於 PCA
    X_other = df_merged[other_genes].copy()
    X_other = X_other.apply(pd.to_numeric, errors="coerce")

    # 刪除有 NaN 的樣本
    valid_mask = X_other.notna().all(axis=1)
    X_other_valid = X_other.loc[valid_mask]

    print(f"  用於 PCA 的樣本數: {len(X_other_valid)}")

    # 執行 PCA
    pca = PCA(n_components=N_PC, random_state=42)
    pcs = pca.fit_transform(X_other_valid.values)

    # 建立 PC DataFrame
    pc_cols = [f"PC{i+1}" for i in range(N_PC)]
    df_pca = pd.DataFrame(pcs, columns=pc_cols, index=X_other_valid.index)

    print(f"  PCA 完成: {df_pca.shape}")
    print(f"  解釋變異量: {pca.explained_variance_ratio_.sum():.4f}")

    # 合併選擇的基因、PCA 成分和臨床特徵
    df_with_pca = df_merged.loc[valid_mask, ["Patient Identifier", col_age, "Stage_Group", col_dfs, col_os] + selected_genes].copy()
    df_with_pca = pd.concat([df_with_pca, df_pca], axis=1)

    print(f"  合併後: {df_with_pca.shape}")

    # 生成 DFS 資料集
    df_dfs = df_with_pca[["Patient Identifier", col_dfs, col_age, "Stage_Group"] + selected_genes + pc_cols].copy()

    # 刪除 DFS 無效的樣本
    dfs_valid = pd.to_numeric(df_dfs[col_dfs], errors="coerce")
    dfs_valid_mask = dfs_valid.notna() & (dfs_valid >= 0)
    df_dfs = df_dfs.loc[dfs_valid_mask].copy()

    dfs_path = os.path.join(folder_name, "dfs_merged_with_clinical_pca.csv")
    df_dfs.to_csv(dfs_path, index=False)
    print(f"  ✓ DFS: {df_dfs.shape} -> {dfs_path}")

    # 生成 OS 資料集
    df_os = df_with_pca[["Patient Identifier", col_os, col_age, "Stage_Group"] + selected_genes + pc_cols].copy()

    # 刪除 OS 無效的樣本
    os_valid = pd.to_numeric(df_os[col_os], errors="coerce")
    os_valid_mask = os_valid.notna() & (os_valid >= 0)
    df_os = df_os.loc[os_valid_mask].copy()

    os_path = os.path.join(folder_name, "os_merged_with_clinical_pca.csv")
    df_os.to_csv(os_path, index=False)
    print(f"  ✓ OS: {df_os.shape} -> {os_path}")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("預處理完成！")
print("=" * 80)

print("\n生成的檔案:")
print("  cleaning_log.csv - 清理步驟記錄")
for folder_name in folder_names:
    print(f"\n  {folder_name}/")
    print(f"    - dfs_merged_with_clinical_pca.csv")
    print(f"    - os_merged_with_clinical_pca.csv")

print("\n共生成 8 個訓練用 CSV 檔案")
print("\n每個 CSV 包含:")
print("  - Patient Identifier")
print("  - Target Variable (DFS 或 OS)")
print("  - Diagnosis Age")
print("  - Stage_Group (1-4)")
print("  - Selected Genes (依基因集而定)")
print("  - PC1-PC200")

print("\n準備進行模型訓練！")
