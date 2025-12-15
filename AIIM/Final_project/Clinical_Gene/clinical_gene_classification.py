"""
Clinical_Gene 資料集的二值化分類訓練
包含臨床特徵 (Age, Stage_Group) + 基因特徵 + PCA 特徵
"""

import os
import numpy as np
import pandas as pd
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
print("開始 Clinical_Gene 二值化分類訓練")
print("=" * 80)

# ============================================================================
# 訓練配置
# ============================================================================
PC_COUNTS = [0, 10, 20, 30, 40, 50, 80, 100, 150, 200]
YEARS = list(range(1, 11))  # 1~10 year

GENE_SETS = [
    "folder1_pam50",
    "folder2_oncotype21",
    "folder3_union_pam50_oncotype",
    "folder4_intersection_pam50_oncotype"
]

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

    # 臨床特徵
    clinical_cols = ["Diagnosis Age", "Stage_Group"]

    # PC 特徵
    pc_cols_all = sorted([c for c in df.columns if c.startswith("PC")],
                         key=lambda x: int(x.replace("PC", "")))

    # 基因特徵（不包含 meta, clinical, PC）
    gene_cols = [c for c in df.columns
                 if c not in meta_cols and c not in clinical_cols and c not in pc_cols_all]

    print(f"\n{'='*60}")
    print(f"任務: {task_name}")
    print(f"臨床特徵數: {len(clinical_cols)}")
    print(f"基因特徵數: {len(gene_cols)}")
    print(f"PC特徵數: {len(pc_cols_all)}")
    print(f"{'='*60}")

    months = pd.to_numeric(df[target_col], errors="coerce")
    df = df.copy()
    df[target_col] = months

    # 轉換特徵為數值
    for c in clinical_cols + gene_cols + pc_cols_all:
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

            # 特徵組成：臨床特徵 + 基因特徵 + PC特徵
            feature_cols = clinical_cols + gene_cols + cur_pc_cols

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

# ============================================================================
# 主訓練流程
# ============================================================================
print("\n開始訓練所有資料集...")
print(f"總配置數: {len(GENE_SETS)} × 2 (DFS/OS) × {len(YEARS)} 年 × {len(PC_COUNTS)} PC × 7 模型")
print(f"         = {len(GENE_SETS) * 2 * len(YEARS) * len(PC_COUNTS) * 7} 個配置")

for folder_name in GENE_SETS:
    print(f"\n{'='*80}")
    print(f"處理資料夾: {folder_name}")
    print(f"{'='*80}")

    # DFS
    dfs_path = f"{folder_name}/dfs_merged_with_clinical_pca.csv"
    if os.path.exists(dfs_path):
        df_dfs = pd.read_csv(dfs_path)
        dfs_results = run_classification_grid(
            df_dfs, "Disease Free (Months)", f"{folder_name} DFS"
        )
        result_path = f"{folder_name}/dfs_classification_results.csv"
        dfs_results.to_csv(result_path, index=False)
        print(f"✓ DFS 結果已儲存: {result_path}")
        print(f"  共 {len(dfs_results)} 個配置")
    else:
        print(f"✗ 檔案不存在: {dfs_path}")

    # OS
    os_path = f"{folder_name}/os_merged_with_clinical_pca.csv"
    if os.path.exists(os_path):
        df_os = pd.read_csv(os_path)
        os_results = run_classification_grid(
            df_os, "Overall Survival (Months)", f"{folder_name} OS"
        )
        result_path = f"{folder_name}/os_classification_results.csv"
        os_results.to_csv(result_path, index=False)
        print(f"✓ OS 結果已儲存: {result_path}")
        print(f"  共 {len(os_results)} 個配置")
    else:
        print(f"✗ 檔案不存在: {os_path}")

print("\n" + "=" * 80)
print("全部完成！")
print("=" * 80)
print("\n生成的檔案：")
for folder in GENE_SETS:
    print(f"  {folder}/")
    print(f"    - dfs_classification_results.csv")
    print(f"    - os_classification_results.csv")
