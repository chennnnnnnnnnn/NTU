"""
綜合分析 6 個資料夾的分類結果
- 舊版 (無右設限): Clinical, Gene, Clinical_Gene
- 新版 (有右設限): Clinical2, Gene2, Clinical_Gene_2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

print("=" * 80)
print("綜合分析 6 個資料夾的分類結果")
print("=" * 80)

# ============================================================================
# 載入所有結果
# ============================================================================
print("\n[步驟 1] 載入所有結果...")

results = {}

# 舊版 - Clinical (無右設限)
try:
    clinical_dfs = pd.read_csv('../Clinical/dfs_binary_results_by_year_balanced_with_thresholds.csv')
    clinical_os = pd.read_csv('../Clinical/os_binary_results_by_year_balanced_with_thresholds.csv')
    # 統一欄位名稱
    clinical_dfs = clinical_dfs.rename(columns={'Year': 'year', 'Model': 'model', 'ROC_AUC': 'roc_auc'})
    clinical_os = clinical_os.rename(columns={'Year': 'year', 'Model': 'model', 'ROC_AUC': 'roc_auc'})
    results['Clinical'] = {'dfs': clinical_dfs, 'os': clinical_os, 'version': 'old', 'censoring': False}
    print(f"  Clinical: DFS {len(clinical_dfs)} rows, OS {len(clinical_os)} rows")
except Exception as e:
    print(f"  Clinical: 載入失敗 - {e}")

# 舊版 - Gene (無右設限) - 使用 intersection
try:
    gene_dfs = pd.read_csv('../Gene/folder4_intersection_pam50_oncotype/dfs_classification_results.csv')
    gene_os = pd.read_csv('../Gene/folder4_intersection_pam50_oncotype/os_classification_results.csv')
    results['Gene'] = {'dfs': gene_dfs, 'os': gene_os, 'version': 'old', 'censoring': False}
    print(f"  Gene: DFS {len(gene_dfs)} rows, OS {len(gene_os)} rows")
except Exception as e:
    print(f"  Gene: 載入失敗 - {e}")

# 舊版 - Clinical_Gene (無右設限) - 使用 intersection
try:
    cg_dfs = pd.read_csv('../Clinical_Gene/folder4_intersection_pam50_oncotype/dfs_classification_results.csv')
    cg_os = pd.read_csv('../Clinical_Gene/folder4_intersection_pam50_oncotype/os_classification_results.csv')
    results['Clinical_Gene'] = {'dfs': cg_dfs, 'os': cg_os, 'version': 'old', 'censoring': False}
    print(f"  Clinical_Gene: DFS {len(cg_dfs)} rows, OS {len(cg_os)} rows")
except Exception as e:
    print(f"  Clinical_Gene: 載入失敗 - {e}")

# 新版 - Clinical2 (有右設限)
try:
    clinical2_dfs = pd.read_csv('../Clinical2/dfs_results_censored.csv')
    clinical2_os = pd.read_csv('../Clinical2/os_results_censored.csv')
    results['Clinical2'] = {'dfs': clinical2_dfs, 'os': clinical2_os, 'version': 'new', 'censoring': True}
    print(f"  Clinical2: DFS {len(clinical2_dfs)} rows, OS {len(clinical2_os)} rows")
except Exception as e:
    print(f"  Clinical2: 載入失敗 - {e}")

# 新版 - Gene2 (有右設限) - 使用 intersection
try:
    gene2_dfs = pd.read_csv('../Gene2/dfs_intersection_results_censored.csv')
    gene2_os = pd.read_csv('../Gene2/os_intersection_results_censored.csv')
    results['Gene2'] = {'dfs': gene2_dfs, 'os': gene2_os, 'version': 'new', 'censoring': True}
    print(f"  Gene2: DFS {len(gene2_dfs)} rows, OS {len(gene2_os)} rows")
except Exception as e:
    print(f"  Gene2: 載入失敗 - {e}")

# 新版 - Clinical_Gene_2 (有右設限) - 使用 intersection
try:
    cg2_dfs = pd.read_csv('../Clinical_Gene_2/dfs_intersection_results_censored.csv')
    cg2_os = pd.read_csv('../Clinical_Gene_2/os_intersection_results_censored.csv')
    results['Clinical_Gene_2'] = {'dfs': cg2_dfs, 'os': cg2_os, 'version': 'new', 'censoring': True}
    print(f"  Clinical_Gene_2: DFS {len(cg2_dfs)} rows, OS {len(cg2_os)} rows")
except Exception as e:
    print(f"  Clinical_Gene_2: 載入失敗 - {e}")

# ============================================================================
# 計算統計
# ============================================================================
print("\n[步驟 2] 計算統計...")

def get_stats(df):
    """計算 AUC 統計"""
    if 'roc_auc' not in df.columns:
        return None
    auc = df['roc_auc'].dropna()
    return {
        'mean': auc.mean(),
        'std': auc.std(),
        'max': auc.max(),
        'min': auc.min(),
        'best_year': df.loc[df['roc_auc'].idxmax(), 'year'] if len(df) > 0 else None
    }

def get_best_auc_by_year(df):
    """取得各年份最佳 AUC"""
    if 'roc_auc' not in df.columns:
        return pd.Series()
    return df.groupby('year')['roc_auc'].max()

# 統計表
stats_data = []
for name, data in results.items():
    dfs_stats = get_stats(data['dfs'])
    os_stats = get_stats(data['os'])
    if dfs_stats and os_stats:
        stats_data.append({
            'Dataset': name,
            'Version': 'New (Censored)' if data['censoring'] else 'Old (No Censoring)',
            'DFS_Mean_AUC': dfs_stats['mean'],
            'DFS_Max_AUC': dfs_stats['max'],
            'DFS_Best_Year': dfs_stats['best_year'],
            'OS_Mean_AUC': os_stats['mean'],
            'OS_Max_AUC': os_stats['max'],
            'OS_Best_Year': os_stats['best_year'],
        })

df_stats = pd.DataFrame(stats_data)
df_stats.to_csv('all_datasets_statistics.csv', index=False)
print("  已儲存: all_datasets_statistics.csv")

# ============================================================================
# 圖1: 6個資料夾 Best AUC 比較
# ============================================================================
print("\n[步驟 3] 生成比較圖表...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 顏色設定
colors_old = {'Clinical': '#1f77b4', 'Gene': '#2ca02c', 'Clinical_Gene': '#d62728'}
colors_new = {'Clinical2': '#17becf', 'Gene2': '#98df8a', 'Clinical_Gene_2': '#ff9896'}

# DFS - 所有資料夾
ax1 = axes[0, 0]
for name, data in results.items():
    best_auc = get_best_auc_by_year(data['dfs'])
    if len(best_auc) > 0:
        color = colors_old.get(name, colors_new.get(name, 'gray'))
        linestyle = '--' if data['version'] == 'old' else '-'
        marker = 'o' if data['version'] == 'old' else 's'
        ax1.plot(best_auc.index, best_auc.values, f'{marker}{linestyle}',
                label=f"{name} ({'舊' if data['version']=='old' else '新'})",
                color=color, linewidth=2, markersize=6)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Best AUC', fontsize=12)
ax1.set_title('DFS: All 6 Datasets Comparison', fontsize=14)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 11))
ax1.set_ylim(0.5, 0.85)

# OS - 所有資料夾
ax2 = axes[0, 1]
for name, data in results.items():
    best_auc = get_best_auc_by_year(data['os'])
    if len(best_auc) > 0:
        color = colors_old.get(name, colors_new.get(name, 'gray'))
        linestyle = '--' if data['version'] == 'old' else '-'
        marker = 'o' if data['version'] == 'old' else 's'
        ax2.plot(best_auc.index, best_auc.values, f'{marker}{linestyle}',
                label=f"{name} ({'舊' if data['version']=='old' else '新'})",
                color=color, linewidth=2, markersize=6)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Best AUC', fontsize=12)
ax2.set_title('OS: All 6 Datasets Comparison', fontsize=14)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 11))
ax2.set_ylim(0.5, 0.95)

# 平均 AUC 長條圖 - DFS
ax3 = axes[1, 0]
datasets = list(results.keys())
x = np.arange(len(datasets))
dfs_means = [get_stats(results[d]['dfs'])['mean'] for d in datasets]
colors = [colors_old.get(d, colors_new.get(d, 'gray')) for d in datasets]
bars = ax3.bar(x, dfs_means, color=colors, alpha=0.8, edgecolor='black')
ax3.set_xlabel('Dataset', fontsize=12)
ax3.set_ylabel('Mean AUC', fontsize=12)
ax3.set_title('DFS: Mean AUC by Dataset', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(datasets, rotation=30, ha='right', fontsize=10)
ax3.set_ylim(0.4, 0.7)
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, dfs_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 平均 AUC 長條圖 - OS
ax4 = axes[1, 1]
os_means = [get_stats(results[d]['os'])['mean'] for d in datasets]
bars = ax4.bar(x, os_means, color=colors, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Dataset', fontsize=12)
ax4.set_ylabel('Mean AUC', fontsize=12)
ax4.set_title('OS: Mean AUC by Dataset', fontsize=14)
ax4.set_xticks(x)
ax4.set_xticklabels(datasets, rotation=30, ha='right', fontsize=10)
ax4.set_ylim(0.5, 0.8)
ax4.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, os_means):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('all_6_datasets_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已儲存: all_6_datasets_comparison.png")

# ============================================================================
# 圖2: 舊版 vs 新版比較
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

old_datasets = ['Clinical', 'Gene', 'Clinical_Gene']
new_datasets = ['Clinical2', 'Gene2', 'Clinical_Gene_2']

# DFS 舊版
ax1 = axes[0, 0]
for name in old_datasets:
    if name in results:
        best_auc = get_best_auc_by_year(results[name]['dfs'])
        ax1.plot(best_auc.index, best_auc.values, 'o--', label=name, linewidth=2, markersize=6)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Best AUC', fontsize=12)
ax1.set_title('DFS: Old Version (No Censoring)', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 11))
ax1.set_ylim(0.5, 0.85)

# DFS 新版
ax2 = axes[0, 1]
for name in new_datasets:
    if name in results:
        best_auc = get_best_auc_by_year(results[name]['dfs'])
        ax2.plot(best_auc.index, best_auc.values, 's-', label=name, linewidth=2, markersize=6)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Best AUC', fontsize=12)
ax2.set_title('DFS: New Version (With Censoring)', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, 11))
ax2.set_ylim(0.5, 0.85)

# OS 舊版
ax3 = axes[1, 0]
for name in old_datasets:
    if name in results:
        best_auc = get_best_auc_by_year(results[name]['os'])
        ax3.plot(best_auc.index, best_auc.values, 'o--', label=name, linewidth=2, markersize=6)
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Best AUC', fontsize=12)
ax3.set_title('OS: Old Version (No Censoring)', fontsize=14)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(1, 11))
ax3.set_ylim(0.5, 0.95)

# OS 新版
ax4 = axes[1, 1]
for name in new_datasets:
    if name in results:
        best_auc = get_best_auc_by_year(results[name]['os'])
        ax4.plot(best_auc.index, best_auc.values, 's-', label=name, linewidth=2, markersize=6)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Best AUC', fontsize=12)
ax4.set_title('OS: New Version (With Censoring)', fontsize=14)
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 11))
ax4.set_ylim(0.5, 0.95)

plt.tight_layout()
plt.savefig('old_vs_new_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已儲存: old_vs_new_comparison.png")

# ============================================================================
# 圖3: 特徵類型比較 (Clinical vs Gene vs Clinical+Gene)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 舊版比較
ax1 = axes[0]
x = np.arange(3)
width = 0.35
old_dfs = [get_stats(results[d]['dfs'])['mean'] for d in old_datasets if d in results]
old_os = [get_stats(results[d]['os'])['mean'] for d in old_datasets if d in results]
ax1.bar(x - width/2, old_dfs, width, label='DFS', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, old_os, width, label='OS', color='indianred', alpha=0.8)
ax1.set_xlabel('Feature Type', fontsize=12)
ax1.set_ylabel('Mean AUC', fontsize=12)
ax1.set_title('Old Version: Feature Type Comparison', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(['Clinical\n(Age+Stage)', 'Gene\n(11 genes)', 'Clinical+Gene\n(All)'])
ax1.legend()
ax1.set_ylim(0.4, 0.8)
ax1.grid(True, alpha=0.3, axis='y')

# 新版比較
ax2 = axes[1]
new_dfs = [get_stats(results[d]['dfs'])['mean'] for d in new_datasets if d in results]
new_os = [get_stats(results[d]['os'])['mean'] for d in new_datasets if d in results]
ax2.bar(x - width/2, new_dfs, width, label='DFS', color='steelblue', alpha=0.8)
ax2.bar(x + width/2, new_os, width, label='OS', color='indianred', alpha=0.8)
ax2.set_xlabel('Feature Type', fontsize=12)
ax2.set_ylabel('Mean AUC', fontsize=12)
ax2.set_title('New Version: Feature Type Comparison', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(['Clinical2\n(Age+Stage)', 'Gene2\n(11 genes)', 'Clinical_Gene_2\n(All)'])
ax2.legend()
ax2.set_ylim(0.4, 0.8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature_type_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已儲存: feature_type_comparison.png")

# ============================================================================
# 圖4: 舊版 vs 新版 直接對比
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pairs = [('Clinical', 'Clinical2'), ('Gene', 'Gene2'), ('Clinical_Gene', 'Clinical_Gene_2')]
labels = ['Clinical', 'Gene', 'Clinical+Gene']

# DFS
ax1 = axes[0]
x = np.arange(3)
width = 0.35
old_vals = [get_stats(results[p[0]]['dfs'])['mean'] for p in pairs if p[0] in results]
new_vals = [get_stats(results[p[1]]['dfs'])['mean'] for p in pairs if p[1] in results]
bars1 = ax1.bar(x - width/2, old_vals, width, label='Old (No Censoring)', color='gray', alpha=0.7)
bars2 = ax1.bar(x + width/2, new_vals, width, label='New (With Censoring)', color='steelblue', alpha=0.8)
ax1.set_xlabel('Feature Type', fontsize=12)
ax1.set_ylabel('Mean AUC', fontsize=12)
ax1.set_title('DFS: Old vs New Version', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.set_ylim(0.4, 0.7)
ax1.grid(True, alpha=0.3, axis='y')
# 標註差異
for i, (old, new) in enumerate(zip(old_vals, new_vals)):
    diff = (new - old) / old * 100
    color = 'green' if diff > 0 else 'red'
    ax1.annotate(f'{diff:+.1f}%', xy=(i, max(old, new) + 0.01), ha='center', fontsize=10, color=color)

# OS
ax2 = axes[1]
old_vals = [get_stats(results[p[0]]['os'])['mean'] for p in pairs if p[0] in results]
new_vals = [get_stats(results[p[1]]['os'])['mean'] for p in pairs if p[1] in results]
bars1 = ax2.bar(x - width/2, old_vals, width, label='Old (No Censoring)', color='gray', alpha=0.7)
bars2 = ax2.bar(x + width/2, new_vals, width, label='New (With Censoring)', color='indianred', alpha=0.8)
ax2.set_xlabel('Feature Type', fontsize=12)
ax2.set_ylabel('Mean AUC', fontsize=12)
ax2.set_title('OS: Old vs New Version', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.set_ylim(0.5, 0.8)
ax2.grid(True, alpha=0.3, axis='y')
# 標註差異
for i, (old, new) in enumerate(zip(old_vals, new_vals)):
    diff = (new - old) / old * 100
    color = 'green' if diff > 0 else 'red'
    ax2.annotate(f'{diff:+.1f}%', xy=(i, max(old, new) + 0.01), ha='center', fontsize=10, color=color)

plt.tight_layout()
plt.savefig('old_new_direct_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已儲存: old_new_direct_comparison.png")

# ============================================================================
# 輸出統計報告
# ============================================================================
print("\n" + "=" * 80)
print("統計報告")
print("=" * 80)

print("\n【各資料夾 AUC 統計】")
print(df_stats.to_string(index=False))

print("\n【舊版 vs 新版比較】")
print("-" * 60)
for old_name, new_name in pairs:
    if old_name in results and new_name in results:
        old_dfs_mean = get_stats(results[old_name]['dfs'])['mean']
        new_dfs_mean = get_stats(results[new_name]['dfs'])['mean']
        old_os_mean = get_stats(results[old_name]['os'])['mean']
        new_os_mean = get_stats(results[new_name]['os'])['mean']

        dfs_diff = (new_dfs_mean - old_dfs_mean) / old_dfs_mean * 100
        os_diff = (new_os_mean - old_os_mean) / old_os_mean * 100

        print(f"\n{old_name} → {new_name}:")
        print(f"  DFS: {old_dfs_mean:.4f} → {new_dfs_mean:.4f} ({dfs_diff:+.2f}%)")
        print(f"  OS:  {old_os_mean:.4f} → {new_os_mean:.4f} ({os_diff:+.2f}%)")

print("\n【最佳配置】")
print("-" * 60)
# 找出 DFS 最佳
dfs_best = max([(name, get_stats(data['dfs'])['mean']) for name, data in results.items()], key=lambda x: x[1])
os_best = max([(name, get_stats(data['os'])['mean']) for name, data in results.items()], key=lambda x: x[1])
print(f"DFS 最佳: {dfs_best[0]} (Mean AUC = {dfs_best[1]:.4f})")
print(f"OS 最佳:  {os_best[0]} (Mean AUC = {os_best[1]:.4f})")

# 新版中最佳
new_dfs_best = max([(name, get_stats(data['dfs'])['mean']) for name, data in results.items() if data['version'] == 'new'], key=lambda x: x[1])
new_os_best = max([(name, get_stats(data['os'])['mean']) for name, data in results.items() if data['version'] == 'new'], key=lambda x: x[1])
print(f"\n新版中 DFS 最佳: {new_dfs_best[0]} (Mean AUC = {new_dfs_best[1]:.4f})")
print(f"新版中 OS 最佳:  {new_os_best[0]} (Mean AUC = {new_os_best[1]:.4f})")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
print("\n生成的檔案:")
print("  - all_datasets_statistics.csv")
print("  - all_6_datasets_comparison.png")
print("  - old_vs_new_comparison.png")
print("  - feature_type_comparison.png")
print("  - old_new_direct_comparison.png")
