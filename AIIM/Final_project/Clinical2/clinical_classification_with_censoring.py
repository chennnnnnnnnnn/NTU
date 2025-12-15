"""
Clinical2 資料集的二值化分類訓練（含右設限處理）
只使用臨床特徵 (Age, Stage_Group)
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
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Clinical2 二值化分類訓練（含右設限處理）")
print("=" * 80)

# ============================================================================
# 步驟 1: 載入臨床資料
# ============================================================================
print("\n[步驟 1] 載入臨床資料...")
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

print(f"原始臨床資料: {len(df_clinical)} 筆")

# ============================================================================
# 步驟 2: 資料清理
# ============================================================================
print("\n[步驟 2] 資料清理...")

# 刪除男性
df_clinical = df_clinical[df_clinical[col_sex] == "Female"].copy()
print(f"刪除男性後: {len(df_clinical)} 筆")

# 刪除年齡缺失
df_clinical = df_clinical[df_clinical[col_age].notna()].copy()
print(f"刪除年齡缺失後: {len(df_clinical)} 筆")

# Stage 處理
stage_map = {
    'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
    'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
    'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
    'Stage IV': 4,
    'Stage X': np.nan
}
df_clinical['Stage_Group'] = df_clinical[col_stage].map(stage_map)
df_clinical = df_clinical[df_clinical['Stage_Group'].notna()].copy()
print(f"刪除 Stage 問題後: {len(df_clinical)} 筆")

# Status 統計
print(f"\nStatus 欄位值:")
print(f"  OS: {df_clinical[col_os_status].value_counts().to_dict()}")
print(f"  DFS: {df_clinical[col_dfs_status].value_counts().to_dict()}")

# ============================================================================
# 步驟 3: 準備 DFS 和 OS 資料
# ============================================================================
print("\n[步驟 3] 準備 DFS 和 OS 資料...")

# DFS 資料
df_dfs = df_clinical[['Patient Identifier', col_age, 'Stage_Group', col_dfs, col_dfs_status]].copy()
df_dfs = df_dfs[df_dfs[col_dfs].notna() & (df_dfs[col_dfs] >= 0)].copy()
df_dfs.to_csv('dfs_with_status.csv', index=False)
print(f"DFS 有效樣本: {len(df_dfs)} 筆")

# OS 資料
df_os = df_clinical[['Patient Identifier', col_age, 'Stage_Group', col_os, col_os_status]].copy()
df_os = df_os[df_os[col_os].notna() & (df_os[col_os] >= 0)].copy()
df_os.to_csv('os_with_status.csv', index=False)
print(f"OS 有效樣本: {len(df_os)} 筆")

# ============================================================================
# 步驟 4: 分類器定義
# ============================================================================
YEARS = list(range(1, 11))
PC_COUNTS = [0]  # 臨床資料沒有 PCA

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

# ============================================================================
# 步驟 5: 分類訓練（含右設限處理）
# ============================================================================
def run_classification_with_censoring(df, target_col, status_col, task_name, censored_value):
    """對不同年份 cutoff 進行二值化分類（含右設限處理）"""
    print(f"\n{'='*60}")
    print(f"任務: {task_name}")
    print(f"{'='*60}")

    feature_cols = ['Diagnosis Age', 'Stage_Group']

    results = []
    sample_counts = []
    classifiers = get_classifiers()

    for year in YEARS:
        cutoff_months = year * 12

        # 建立二值標籤
        df_year = df.copy()
        df_year['label'] = (df_year[target_col] >= cutoff_months).astype(int)

        # 右設限處理：刪除 time < cutoff 且 status = censored_value 的樣本
        censored_mask = (df_year[target_col] < cutoff_months) & (df_year[status_col] == censored_value)
        n_censored = censored_mask.sum()
        df_year = df_year[~censored_mask].copy()

        n_total = len(df_year)
        n_pos = (df_year['label'] == 1).sum()
        n_neg = (df_year['label'] == 0).sum()

        print(f"\n---- Year >= {year} (樣本:{n_total}, 正:{n_pos}, 負:{n_neg}, 刪除右設限:{n_censored}) ----")

        sample_counts.append({
            'Year': year,
            'Total': n_total,
            'Positive': n_pos,
            'Negative': n_neg,
            'Censored_removed': n_censored
        })

        if n_total < 20 or n_pos < 5 or n_neg < 5:
            print("  樣本太少，跳過")
            continue

        # 準備特徵和標籤
        X = df_year[feature_cols].values.astype(float)
        y = df_year['label'].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_auc = 0
        best_model = ""

        for model_name, model in classifiers.items():
            y_true_all = []
            y_pred_all = []
            y_score_all = []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_all.append(y_pred)
                y_true_all.append(y_test)

                # 預測分數
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

            # 計算指標
            acc = accuracy_score(y_true_all, y_pred_all)
            prec = precision_score(y_true_all, y_pred_all, zero_division=0)
            rec = recall_score(y_true_all, y_pred_all, zero_division=0)
            f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

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

            # 掃描閾值
            best_f1 = f1
            best_f1_th = np.nan
            best_J = np.nan
            best_J_th = np.nan

            if y_score_all is not None:
                for th in np.linspace(0.05, 0.95, 19):
                    y_pred_th = (y_score_all >= th).astype(int)
                    f1_th = f1_score(y_true_all, y_pred_th, zero_division=0)
                    if f1_th > best_f1:
                        best_f1 = f1_th
                        best_f1_th = th

                    J, _, _ = compute_youden(y_true_all, y_pred_th)
                    if np.isnan(best_J) or J > best_J:
                        best_J = J
                        best_J_th = th

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_model = model_name

            results.append({
                'year': year,
                'model': model_name,
                'n_samples': n_total,
                'n_positive': n_pos,
                'n_negative': n_neg,
                'n_censored_removed': n_censored,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': auc,
                'best_f1': best_f1,
                'best_f1_threshold': best_f1_th,
                'best_youden': best_J,
                'best_youden_threshold': best_J_th,
            })

        if best_model:
            print(f"  Best: {best_model} AUC={best_auc:.3f}")

    return pd.DataFrame(results), pd.DataFrame(sample_counts)

# ============================================================================
# 執行分類
# ============================================================================
print("\n[步驟 5] 執行分類訓練...")

# DFS
dfs_results, dfs_counts = run_classification_with_censoring(
    df_dfs, col_dfs, col_dfs_status, "DFS Classification", "0:DiseaseFree"
)
dfs_results.to_csv('dfs_results_censored.csv', index=False)
dfs_counts.to_csv('dfs_sample_counts.csv', index=False)
print(f"\n已儲存: dfs_results_censored.csv, dfs_sample_counts.csv")

# OS
os_results, os_counts = run_classification_with_censoring(
    df_os, col_os, col_os_status, "OS Classification", "0:LIVING"
)
os_results.to_csv('os_results_censored.csv', index=False)
os_counts.to_csv('os_sample_counts.csv', index=False)
print(f"已儲存: os_results_censored.csv, os_sample_counts.csv")

# ============================================================================
# 步驟 6: 畫圖
# ============================================================================
print("\n[步驟 6] 生成比較圖...")

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['figure.figsize'] = (14, 10)

# 圖1: AUC 比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# DFS 各模型 AUC by Year
ax1 = axes[0, 0]
models = dfs_results['model'].unique()
for model in models:
    model_data = dfs_results[dfs_results['model'] == model]
    ax1.plot(model_data['year'], model_data['roc_auc'], 'o-', label=model, linewidth=2, markersize=5)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('AUC', fontsize=12)
ax1.set_title('DFS: AUC by Year and Model', fontsize=14)
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 11))
ax1.set_ylim(0.4, 0.9)

# OS 各模型 AUC by Year
ax2 = axes[0, 1]
for model in models:
    model_data = os_results[os_results['model'] == model]
    ax2.plot(model_data['year'], model_data['roc_auc'], 'o-', label=model, linewidth=2, markersize=5)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('AUC', fontsize=12)
ax2.set_title('OS: AUC by Year and Model', fontsize=14)
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 11))
ax2.set_ylim(0.4, 0.9)

# DFS 最佳 AUC
ax3 = axes[1, 0]
best_dfs = dfs_results.groupby('year')['roc_auc'].max()
best_os = os_results.groupby('year')['roc_auc'].max()
ax3.plot(best_dfs.index, best_dfs.values, 'o-', label='DFS', linewidth=2, markersize=8, color='blue')
ax3.plot(best_os.index, best_os.values, 's-', label='OS', linewidth=2, markersize=8, color='red')
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Best AUC', fontsize=12)
ax3.set_title('Best AUC: DFS vs OS', fontsize=14)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 11))
ax3.set_ylim(0.4, 0.9)

# 各模型平均 AUC
ax4 = axes[1, 1]
x = np.arange(len(models))
width = 0.35
dfs_means = [dfs_results[dfs_results['model'] == m]['roc_auc'].mean() for m in models]
os_means = [os_results[os_results['model'] == m]['roc_auc'].mean() for m in models]
ax4.bar(x - width/2, dfs_means, width, label='DFS', color='blue', alpha=0.7)
ax4.bar(x + width/2, os_means, width, label='OS', color='red', alpha=0.7)
ax4.set_xlabel('Model', fontsize=12)
ax4.set_ylabel('Mean AUC', fontsize=12)
ax4.set_title('Mean AUC by Model', fontsize=14)
ax4.set_xticks(x)
ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0.4, 0.8)

plt.tight_layout()
plt.savefig('clinical_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: clinical_auc_comparison.png")

# 圖2: 熱力圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# DFS 熱力圖
dfs_pivot = dfs_results.pivot(index='model', columns='year', values='roc_auc')
im1 = axes[0].imshow(dfs_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.8)
axes[0].set_xticks(range(10))
axes[0].set_xticklabels(range(1, 11))
axes[0].set_yticks(range(len(models)))
axes[0].set_yticklabels(dfs_pivot.index, fontsize=9)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_title('DFS: AUC Heatmap', fontsize=14)
for i in range(len(models)):
    for j in range(10):
        val = dfs_pivot.values[i, j]
        if not np.isnan(val):
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# OS 熱力圖
os_pivot = os_results.pivot(index='model', columns='year', values='roc_auc')
im2 = axes[1].imshow(os_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.8)
axes[1].set_xticks(range(10))
axes[1].set_xticklabels(range(1, 11))
axes[1].set_yticks(range(len(models)))
axes[1].set_yticklabels(os_pivot.index, fontsize=9)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_title('OS: AUC Heatmap', fontsize=14)
for i in range(len(models)):
    for j in range(10):
        val = os_pivot.values[i, j]
        if not np.isnan(val):
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.savefig('clinical_auc_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: clinical_auc_heatmap.png")

# 圖3: 樣本數分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.bar(dfs_counts['Year'] - 0.2, dfs_counts['Positive'], 0.4, label='Positive (>=N years)', color='green', alpha=0.7)
ax1.bar(dfs_counts['Year'] + 0.2, dfs_counts['Negative'], 0.4, label='Negative (<N years)', color='red', alpha=0.7)
ax1.plot(dfs_counts['Year'], dfs_counts['Censored_removed'], 'k--o', label='Censored removed', linewidth=2)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Sample Count', fontsize=12)
ax1.set_title('DFS: Sample Distribution by Year', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 11))

ax2 = axes[1]
ax2.bar(os_counts['Year'] - 0.2, os_counts['Positive'], 0.4, label='Positive (>=N years)', color='green', alpha=0.7)
ax2.bar(os_counts['Year'] + 0.2, os_counts['Negative'], 0.4, label='Negative (<N years)', color='red', alpha=0.7)
ax2.plot(os_counts['Year'], os_counts['Censored_removed'], 'k--o', label='Censored removed', linewidth=2)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Sample Count', fontsize=12)
ax2.set_title('OS: Sample Distribution by Year', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig('clinical_sample_counts.png', dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: clinical_sample_counts.png")

# ============================================================================
# 輸出統計摘要
# ============================================================================
print("\n" + "="*60)
print("AUC 統計摘要")
print("="*60)

print("\n【DFS】")
print(f"  平均 AUC: {dfs_results['roc_auc'].mean():.4f}")
print(f"  最大 AUC: {dfs_results['roc_auc'].max():.4f}")
best_dfs_model = dfs_results.groupby('model')['roc_auc'].mean().idxmax()
print(f"  最佳模型: {best_dfs_model}")

print("\n【OS】")
print(f"  平均 AUC: {os_results['roc_auc'].mean():.4f}")
print(f"  最大 AUC: {os_results['roc_auc'].max():.4f}")
best_os_model = os_results.groupby('model')['roc_auc'].mean().idxmax()
print(f"  最佳模型: {best_os_model}")

print("\n" + "="*60)
print("全部完成！")
print("="*60)
print("\n輸出檔案:")
print("  dfs_with_status.csv")
print("  dfs_results_censored.csv")
print("  dfs_sample_counts.csv")
print("  os_with_status.csv")
print("  os_results_censored.csv")
print("  os_sample_counts.csv")
print("  clinical_auc_comparison.png")
print("  clinical_auc_heatmap.png")
print("  clinical_sample_counts.png")
