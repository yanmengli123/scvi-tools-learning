"""
================================================================================
模块6：贝叶斯差异表达分析 + Marker 绘图
================================================================================
【scVI 差异表达 vs 传统 DE 的区别】
    传统 DE (Wilcoxon / t-test):
      - 直接比较两组细胞的 count 数
      - 受批次/技术噪声影响大
      - 假设正态/对称分布

    scVI 差异表达:
      - 在隐空间 (Z) 比较两组的分布
      - 通过解码器生成"假设表达"再比较
      - 用贝叶斯框架：proba_de 表示"该基因在两组间有差异的后验概率"
      - 给出 raw_mean1, raw_mean2, lfc, bayes_factor 等多种统计量

【关键输出列】
    proba_de / proba_m1: 差异表达的后验概率 (0-1)
    bayes_factor:        贝叶斯因子 (差异证据强度)
    lfc_mean:            log fold change 的后验均值
    raw_mean1, raw_mean2: 各组在 group1/group2 的平均表达
    non_zeros_proportion1/2: 各组中非零细胞比例

【1v1 vs 1vall】
    1v1:  A vs B → 找 A 相对于 B 的 marker
    1vall:  A vs Rest → 找 A 的 marker (其他所有)
    用法:
        1vall 适合找每种细胞类型的"特征 marker"
        1v1  适合具体两群之间的差异（如疾病 vs 健康）
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PARENT_DIR, "code_interpretation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 全局设置 ====================
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# scanpy 默认输出到 figures/ 目录，先确保存在
FIGURES_TMP = os.path.join(OUTPUT_DIR, "figures_tmp")
os.makedirs(FIGURES_TMP, exist_ok=True)
sc.settings.figdir = FIGURES_TMP


# ============================================================
# 步骤 6.1：加载数据
# ============================================================
print("=" * 70)
print("模块6：贝叶斯差异表达 + Marker 绘图")
print("=" * 70)

adata = sc.read_h5ad(os.path.join(OUTPUT_DIR, "clustered_adata.h5ad"))
model = scvi.model.SCVI.load(
    os.path.join(OUTPUT_DIR, "scvi_heart_model") + "/",
    adata=adata
)
print(f"  数据加载: {adata.shape}")


# ============================================================
# 步骤 6.2：1vall 差异分析 - 找每种细胞类型的 marker
# ============================================================
print("\n[6.1] 1vall 差异分析: 找每种细胞类型 marker...")

de_results = model.differential_expression(
    groupby="cell_type"
)
# 不指定 group1/group2 → 1vall 模式
# 输出: (n_genes × n_comparisons, 14) 的 DataFrame
# 每个 comparison 形如 "CellTypeA vs Rest"

print(f"  DE 结果形状: {de_results.shape}")
print(f"  列: {de_results.columns.tolist()}")
print(f"  比较 (前5个): {de_results['comparison'].unique()[:5]}")

de_results.to_csv(os.path.join(OUTPUT_DIR, "differential_expression_all.csv"))
print(f"  保存: differential_expression_all.csv")


# ============================================================
# 步骤 6.3：筛选 top marker genes
# ============================================================
print("\n[6.2] 筛选每种细胞类型 top 5 marker genes...")
cell_types = adata.obs['cell_type'].unique().tolist()
top_markers_dict = {}

for ct in cell_types:
    # 取该细胞类型 vs 其余的 DE 结果
    ct_de = de_results[de_results['comparison'] == f"{ct} vs Rest"]
    if len(ct_de) == 0:
        continue

    # 计算 log2 fold change
    if 'raw_normalized_mean1' in ct_de.columns:
        # group1 = ct, group2 = Rest
        mean1 = ct_de['raw_normalized_mean1'].clip(lower=1e-10)
        mean2 = ct_de['raw_normalized_mean2'].clip(lower=1e-10)
        ct_de = ct_de.copy()
        ct_de['lfc'] = np.log2(mean1 / mean2)

        # 按 LFC 和概率筛选（既高表达又显著）
        top_genes = ct_de.sort_values(
            ['lfc', 'proba_m1'],
            ascending=[False, False]
        ).head(5).index.tolist()
    else:
        # 退而求其次：用 bayes_factor
        top_genes = ct_de.sort_values('bayes_factor', ascending=False).head(5).index.tolist()

    top_markers_dict[ct] = top_genes
    print(f"  {ct}: {top_genes[:3]}...")


# ============================================================
# 步骤 6.4：Dotplot - 显示 marker 在各细胞类型的表达
# ============================================================
print("\n[6.3] 绘制 Dotplot ...")
# Dotplot 包含两层信息：
#   颜色（红色强度）= 平均表达量
#   点的大小 = 表达细胞占比

# 选取前 10 种细胞类型和它们的前 3 个 marker
selected_cts = cell_types[:10] if len(cell_types) > 10 else cell_types
all_markers = []
for ct in selected_cts:
    if ct in top_markers_dict:
        all_markers.extend(top_markers_dict[ct][:3])
all_markers = list(dict.fromkeys(all_markers))[:20]  # 去重，最多 20

print(f"  选取 {len(selected_cts)} 种细胞类型, {len(all_markers)} 个 marker")

if len(all_markers) > 0:
    sc.pl.dotplot(
        adata,
        var_names=all_markers,
        groupby='cell_type',
        standard_scale='var',     # 基因标准化，让不同基因可比
        save='_07_dotplot_markers.png',
        show=False
    )
    # scanpy 默认存到 figdir/dotplot_*.png
    import shutil
    src = os.path.join(FIGURES_TMP, "dotplot_07_dotplot_markers.png")
    dst = os.path.join(OUTPUT_DIR, "07_dotplot_markers.png")
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"  保存: {dst}")


# ============================================================
# 步骤 6.5：Heatmap - 显示 marker 表达热图
# ============================================================
print("\n[6.4] 绘制 Heatmap ...")
if len(all_markers) > 0:
    sc.pl.heatmap(
        adata,
        var_names=all_markers,
        groupby='cell_type',
        standard_scale='var',
        show_gene_labels=True,
        save='_08_heatmap_markers.png',
        show=False
    )
    src = os.path.join(FIGURES_TMP, "heatmap_08_heatmap_markers.png")
    dst = os.path.join(OUTPUT_DIR, "08_heatmap_markers.png")
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"  保存: {dst}")


# ============================================================
# 步骤 6.6：堆叠条形图 - 聚类的细胞类型组成
# ============================================================
print("\n[6.5] 绘制聚类-细胞类型堆叠图 ...")
cluster_celltype = pd.crosstab(adata.obs['leiden_scvi'], adata.obs['cell_type'])

fig, ax = plt.subplots(figsize=(12, 5))
cluster_celltype.plot(kind='bar', stacked=True, colormap='tab20', ax=ax, width=0.85)
ax.set_xlabel('scVI Cluster', fontsize=12)
ax.set_ylabel('Number of cells', fontsize=12)
ax.set_title('Cell type composition per scVI cluster', fontsize=13)
ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_cluster_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/09_cluster_composition.png")


# ============================================================
# 步骤 6.7：差异表达 LFC 分布（小提琴图）
# ============================================================
print("\n[6.6] 绘制 LFC 分布 ...")
# 不同 comparison 的 LFC 分布能展示：
#   - 比较的方向（向上 = group1 高表达）
#   - 分布对称性
#   - 是否有离群基因

fig, ax = plt.subplots(figsize=(10, 6))

# 选前 6 个比较（避免太挤）
top_comparisons = de_results['comparison'].value_counts().head(6).index.tolist()
de_subset = de_results[de_results['comparison'].isin(top_comparisons)].copy()

# 计算 LFC
if 'raw_normalized_mean1' in de_subset.columns:
    de_subset['lfc'] = np.log2(
        (de_subset['raw_normalized_mean1'].clip(lower=1e-10)) /
        (de_subset['raw_normalized_mean2'].clip(lower=1e-10))
    )
else:
    de_subset['lfc'] = np.log2(de_subset['scale1'] / de_subset['scale2'])

sns.violinplot(data=de_subset, x='comparison', y='lfc', ax=ax, palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Log2 Fold Change', fontsize=12)
ax.set_title('LFC distribution across top comparisons', fontsize=13)
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No change')
ax.grid(alpha=0.3, axis='y')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_de_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/10_de_violin.png")


# ============================================================
# 步骤 6.8：1v1 火山图 - Ventricular vs Fibroblast
# ============================================================
print("\n[6.7] 1v1 对比: Ventricular_Cardiomyocyte vs Fibroblast...")
# 1v1 用于：疾病 vs 对照、刺激 vs 未刺激、特定两个亚群

de_1v1 = model.differential_expression(
    groupby="cell_type",
    group1="Ventricular_Cardiomyocyte",   # 实验组
    group2="Fibroblast"                   # 对照组
)
de_1v1.to_csv(os.path.join(OUTPUT_DIR, "de_1v1_VC_vs_FB.csv"))
print(f"  DE 结果: {de_1v1.shape}")

# 火山图
fig, ax = plt.subplots(figsize=(8, 6))

# 计算 LFC
if 'raw_normalized_mean1' in de_1v1.columns:
    de_1v1_lfc = np.log2(
        (de_1v1['raw_normalized_mean1'].clip(lower=1e-10)) /
        (de_1v1['raw_normalized_mean2'].clip(lower=1e-10))
    )
else:
    de_1v1_lfc = np.log2(de_1v1['scale1'] / de_1v1['scale2'])

de_1v1['log_p'] = -np.log10(de_1v1['proba_m1'].clip(lower=1e-300))

# 三色：上调 (coral) / 下调 (steelblue) / 不显著 (gray)
colors = np.where(
    (de_1v1_lfc > 1) & (de_1v1['proba_m1'] > 0.8), 'coral',
    np.where((de_1v1_lfc < -1) & (de_1v1['proba_m1'] > 0.8), 'steelblue', 'gray')
)

ax.scatter(de_1v1_lfc, de_1v1['log_p'], c=colors, s=8, alpha=0.6)
ax.set_xlabel('Log2 Fold Change (VC vs FB)', fontsize=12)
ax.set_ylabel('-log10(P(DE))', fontsize=12)
ax.set_title('Volcano: Ventricular Cardiomyocyte vs Fibroblast', fontsize=13)
ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
ax.axvline(1, color='red', linestyle='--', alpha=0.5)
ax.axvline(-1, color='red', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_volcano_1v1.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/11_volcano_1v1.png")

# 打印 top 上下调基因
print(f"\n  Top 5 上调基因 (VC > FB):")
if 'raw_normalized_mean1' in de_1v1.columns:
    top_up = de_1v1.assign(lfc=de_1v1_lfc).sort_values('lfc', ascending=False).head(5)
    for g in top_up.index:
        print(f"    {g}: lfc={de_1v1_lfc[g]:.2f}, p={top_up.loc[g, 'proba_m1']:.3f}")

print(f"\n  Top 5 下调基因 (VC < FB):")
if 'raw_normalized_mean1' in de_1v1.columns:
    top_down = de_1v1.assign(lfc=de_1v1_lfc).sort_values('lfc', ascending=True).head(5)
    for g in top_down.index:
        print(f"    {g}: lfc={de_1v1_lfc[g]:.2f}, p={top_down.loc[g, 'proba_m1']:.3f}")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("全部 6 模块分析完成！")
print("=" * 70)
print(f"\n生成的图片 ({OUTPUT_DIR}/):")
import glob
for f in sorted(glob.glob(f"{OUTPUT_DIR}/*.png")):
    size = os.path.getsize(f) / 1024
    print(f"  {os.path.basename(f)} ({size:.1f} KB)")

print(f"\n=== 最终汇总 ===")
print(f"  总细胞数: {adata.n_obs}")
print(f"  总基因数(HVG): {adata.n_vars}")
print(f"  原始细胞类型: {adata.obs['cell_type'].nunique()}")
print(f"  scVI 聚类数: {adata.obs['leiden_scvi'].nunique()}")
print(f"  隐空间维度: {adata.obsm['X_scVI'].shape[1]}")
print(f"  差异表达基因对: {de_results.shape[0]}")
