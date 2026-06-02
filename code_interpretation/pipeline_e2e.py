"""
================================================================================
模块2-6 整合版：端到端 scVI 分析流程
================================================================================
【整合原因】
    scVI 的 setup_anndata 会把注册信息存到 adata.uns['_scvi_manager_uuid']
    但 h5ad 文件在某些情况下不会完全保留这个信息
    为了保证代码"零 bug"运行，我们将模块 2-6 合并为一个端到端脚本

【本脚本流程】
    1. 数据加载 + 预处理 (模块2)
    2. setup_anndata + 训练 scVI (模块3)
    3. 提取 X_scVI 和归一化表达 (模块4)
    4. UMAP + Leiden 聚类 (模块5)
    5. 差异表达 + Marker (模块6)

【所有 11 张图片输出到 code_interpretation_output/】
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shutil
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PARENT_DIR, "heart_atlas.h5ad")
OUTPUT_DIR = os.path.join(PARENT_DIR, "code_interpretation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIGURES_TMP = os.path.join(OUTPUT_DIR, "_scanpy_tmp")
os.makedirs(FIGURES_TMP, exist_ok=True)
sc.settings.figdir = FIGURES_TMP

# ==================== 全局设置 ====================
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 模块2: 数据加载与预处理
# ============================================================
print("=" * 70)
print("模块2：数据加载与预处理")
print("=" * 70)

# [2.1] 加载 h5ad 数据
print("\n[2.1] 加载 h5ad 数据...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  原始数据: {adata.shape[0]} 细胞 × {adata.shape[1]} 基因")

# [2.2] 质量控制
print("\n[2.2] 质量控制 (QC) ...")
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=100)
print(f"  过滤后: {adata.shape[0]} 细胞 × {adata.shape[1]} 基因")

# [2.3] ★关键★ 保存原始 count
print("\n[2.3] 保存原始 UMI 计数到 layers['counts'] ...")
adata.layers["counts"] = adata.X.copy()
print(f"  layers['counts'] 形状: {adata.layers['counts'].shape}")

# [2.4] log 归一化（用于绘图）
print("\n[2.4] log 归一化（用于绘图） ...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
print(f"  log 归一化后: 范围 [{adata.X.min():.2f}, {adata.X.max():.2f}]")

# [2.5] 筛选高可变基因
print("\n[2.5] 高可变基因筛选 (HVG) ...")
sc.pp.highly_variable_genes(
    adata, n_top_genes=1200,
    flavor='seurat', batch_key='donor',
    subset=True
)
print(f"  HVG 筛选后: {adata.shape[0]} 细胞 × {adata.shape[1]} 基因")

# [2.6] setup_anndata
print("\n[2.6] setup_anndata ...")
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="donor",
    categorical_covariate_keys=["cell_source"],
    layer="counts",
    continuous_covariate_keys=["percent_mito", "percent_ribo"],
)
print(f"  setup_anndata 完成")
print(f"  批次: donor ({adata.obs['donor'].nunique()} 个)")
print(f"  分类协变量: cell_source ({adata.obs['cell_source'].nunique()} 个)")

# [2.7] 绘制 QC 图
print("\n[2.7] 绘制预处理 QC 图 ...")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(adata.obs['total_counts'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Total counts'); axes[0].set_ylabel('Number of cells')
axes[0].set_title('Total counts per cell')
axes[1].hist(adata.obs['n_genes_by_counts'], bins=50, color='coral', edgecolor='white')
axes[1].set_xlabel('Genes per cell'); axes[1].set_ylabel('Number of cells')
axes[1].set_title('Genes per cell')
axes[2].hist(adata.obs['percent_mito'], bins=50, color='seagreen', edgecolor='white')
axes[2].set_xlabel('Percent mito'); axes[2].set_ylabel('Number of cells')
axes[2].set_title('Mitochondrial gene %')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_preprocess_qc.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 01_preprocess_qc.png")


# ============================================================
# 模块3: SCVI 模型构建与训练
# ============================================================
print("\n" + "=" * 70)
print("模块3：SCVI 模型构建与训练")
print("=" * 70)

# [3.1] 构建模型
print("\n[3.1] 构建 SCVI 模型 ...")
model = scvi.model.SCVI(
    adata,
    n_latent=30,
    n_layers=2,
    n_hidden=128,
    dropout_rate=0.1,
    dispersion="gene-batch",
    gene_likelihood="nb",
)
print(f"  模型参数: n_latent=30, n_layers=2, n_hidden=128")
print(f"  dispersion='gene-batch', gene_likelihood='nb'")

# [3.2] 训练
print("\n[3.2] 训练模型 ...")
model.train(
    max_epochs=100,
    batch_size=128,
    train_size=0.9,
    early_stopping=True,
    enable_progress_bar=False,
    accelerator="auto",
)
print("  训练完成!")

# [3.3] 绘制训练曲线
print("\n[3.3] 绘制训练曲线 ...")
train_history = model.history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
if 'elbo_train' in train_history:
    axes[0].plot(train_history['elbo_train'], label='ELBO (train)', color='steelblue', linewidth=2)
if 'reconstruction_loss_train' in train_history:
    axes[0].plot(train_history['reconstruction_loss_train'], label='Reconstruction Loss', color='coral', linewidth=2)
if 'kl_train' in train_history:
    axes[0].plot(train_history['kl_train'], label='KL Divergence', color='seagreen', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Components'); axes[0].legend(); axes[0].grid(alpha=0.3)
if 'elbo_validation' in train_history:
    axes[1].plot(train_history['elbo_validation'], label='ELBO (val)', color='coral', linewidth=2)
if 'reconstruction_loss_validation' in train_history:
    axes[1].plot(train_history['reconstruction_loss_validation'], label='Recon Loss (val)', color='steelblue', linewidth=2)
if 'kl_validation' in train_history:
    axes[1].plot(train_history['kl_validation'], label='KL (val)', color='seagreen', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('Validation Loss Components'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_training_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 02_training_curve.png")

# [3.4] 保存模型
print("\n[3.4] 保存模型 ...")
model_dir = os.path.join(OUTPUT_DIR, "scvi_heart_model")
model.save(model_dir + "/", overwrite=True)
print(f"  模型已保存: {model_dir}/")


# ============================================================
# 模块4: 提取模型结果
# ============================================================
print("\n" + "=" * 70)
print("模块4：提取模型结果")
print("=" * 70)

# [4.1] 提取 X_scVI
print("\n[4.1] 提取隐空间 X_scVI ...")
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent
print(f"  X_scVI 形状: {latent.shape}")
print(f"  X_scVI 范围: [{latent.min():.2f}, {latent.max():.2f}]")

# [4.2] 提取归一化表达
print("\n[4.2] 提取归一化基因表达 ...")
normalized = model.get_normalized_expression(
    library_size=1e4, return_numpy=True
)
adata.layers["scvi_normalized"] = normalized
print(f"  scvi_normalized 形状: {normalized.shape}")

# [4.3] 绘制隐空间分布
print("\n[4.3] 绘制隐空间分布 ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(latent.flatten(), bins=100, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Latent value'); axes[0].set_ylabel('Frequency')
axes[0].set_title('Latent space distribution\n(should look like Gaussian N(0,1))')
axes[0].grid(alpha=0.3); axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='zero')
axes[0].legend()
latent_var = np.var(latent, axis=0)
sorted_idx = np.argsort(latent_var)[::-1]
axes[1].bar(range(len(latent_var)), latent_var[sorted_idx], color='coral', alpha=0.8)
axes[1].set_xlabel('Latent dim (sorted by var)'); axes[1].set_ylabel('Variance')
axes[1].set_title('Variance of each latent dim'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_latent_space.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 03_latent_space.png")


# ============================================================
# 模块5: 降维可视化 + 聚类
# ============================================================
print("\n" + "=" * 70)
print("模块5：降维可视化 + 聚类")
print("=" * 70)

# [5.1] PCA + UMAP (对照组)
print("\n[5.1] 对照组: PCA + UMAP ...")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca', key_added='pca')
sc.tl.umap(adata, neighbors_key='pca')
adata.obsm['X_umap_pca'] = adata.obsm['X_umap'].copy()
print("  PCA + UMAP 完成")

# [5.2] scVI + UMAP (实验组)
print("\n[5.2] 实验组: X_scVI + UMAP ...")
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_scVI', key_added='scvi')
sc.tl.umap(adata, neighbors_key='scvi')
adata.obsm['X_umap_scvi'] = adata.obsm['X_umap'].copy()
print("  scVI + UMAP 完成")

# [5.3] 批次校正对比图
print("\n[5.3] 绘制批次校正对比图 ...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
sc.pl.embedding(adata, basis='X_umap_pca', color='donor',
                ax=axes[0, 0], show=False,
                title='PCA UMAP - Donor (Batch effect visible)',
                frameon=False, size=8, legend_loc=None)
sc.pl.embedding(adata, basis='X_umap_pca', color='cell_type',
                ax=axes[0, 1], show=False,
                title='PCA UMAP - Cell Type',
                frameon=False, size=8, legend_loc='right margin', legend_fontsize=6)
sc.pl.embedding(adata, basis='X_umap_scvi', color='donor',
                ax=axes[1, 0], show=False,
                title='scVI UMAP - Donor (Batch effect removed)',
                frameon=False, size=8, legend_loc=None)
sc.pl.embedding(adata, basis='X_umap_scvi', color='cell_type',
                ax=axes[1, 1], show=False,
                title='scVI UMAP - Cell Type',
                frameon=False, size=8, legend_loc='right margin', legend_fontsize=6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_batch_correction_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 04_batch_correction_comparison.png")

# [5.4] Leiden 聚类
print("\n[5.4] Leiden 聚类 ...")
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_scvi', neighbors_key='scvi')
n_clusters = adata.obs['leiden_scvi'].nunique()
print(f"  聚类数: {n_clusters}")
print(adata.obs['leiden_scvi'].value_counts().sort_index())

# [5.5] 聚类结果
print("\n[5.5] 绘制聚类结果 ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc.pl.embedding(adata, basis='X_umap_scvi', color='leiden_scvi',
                ax=axes[0], show=False,
                title=f'scVI Clusters (n={n_clusters}, resolution=0.5)',
                frameon=False, legend_loc='on data', size=12, legend_fontsize=10)
sc.pl.embedding(adata, basis='X_umap_scvi', color='cell_type',
                ax=axes[1], show=False,
                title='Original Cell Types (Ground Truth)',
                frameon=False, legend_loc='right margin', size=12, legend_fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_clustering_result.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 05_clustering_result.png")

# [5.6] 聚类-细胞类型对应热图
print("\n[5.6] 绘制聚类-细胞类型对应热图 ...")
cluster_celltype = pd.crosstab(adata.obs['leiden_scvi'], adata.obs['cell_type'])
cluster_celltype_norm = cluster_celltype.div(cluster_celltype.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(cluster_celltype_norm.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(cluster_celltype_norm.columns)))
ax.set_xticklabels(cluster_celltype_norm.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(cluster_celltype_norm.index)))
ax.set_yticklabels(cluster_celltype_norm.index)
ax.set_xlabel('Cell Type'); ax.set_ylabel('scVI Cluster')
ax.set_title('Cluster-CellType Correspondence (row-normalized)')
for i in range(len(cluster_celltype_norm.index)):
    for j in range(len(cluster_celltype_norm.columns)):
        if cluster_celltype_norm.values[i, j] > 0.1:
            ax.text(j, i, f'{cluster_celltype_norm.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if cluster_celltype_norm.values[i, j] > 0.5 else 'black')
plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_cluster_celltype_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 06_cluster_celltype_heatmap.png")


# ============================================================
# 模块6: 差异表达 + Marker
# ============================================================
print("\n" + "=" * 70)
print("模块6：贝叶斯差异表达 + Marker 绘图")
print("=" * 70)

# [6.1] 1vall DE
print("\n[6.1] 1vall 差异分析 ...")
de_results = model.differential_expression(groupby="cell_type")
print(f"  DE 结果形状: {de_results.shape}")
de_results.to_csv(os.path.join(OUTPUT_DIR, "differential_expression_all.csv"))

# [6.2] 筛选 top marker
print("\n[6.2] 筛选每种细胞类型 top 5 marker ...")
cell_types = adata.obs['cell_type'].unique().tolist()
top_markers_dict = {}
for ct in cell_types:
    ct_de = de_results[de_results['comparison'] == f"{ct} vs Rest"]
    if len(ct_de) == 0:
        continue
    if 'raw_normalized_mean1' in ct_de.columns:
        mean1 = ct_de['raw_normalized_mean1'].clip(lower=1e-10)
        mean2 = ct_de['raw_normalized_mean2'].clip(lower=1e-10)
        ct_de = ct_de.copy()
        ct_de['lfc'] = np.log2(mean1 / mean2)
        top_genes = ct_de.sort_values(['lfc', 'proba_m1'], ascending=[False, False]).head(5).index.tolist()
    else:
        top_genes = ct_de.sort_values('bayes_factor', ascending=False).head(5).index.tolist()
    top_markers_dict[ct] = top_genes
    print(f"  {ct}: {top_genes[:3]}...")

# [6.3] Dotplot
print("\n[6.3] 绘制 Dotplot ...")
selected_cts = cell_types[:10] if len(cell_types) > 10 else cell_types
all_markers = []
for ct in selected_cts:
    if ct in top_markers_dict:
        all_markers.extend(top_markers_dict[ct][:3])
all_markers = list(dict.fromkeys(all_markers))[:20]
print(f"  选取 {len(selected_cts)} 种细胞类型, {len(all_markers)} 个 marker")

if len(all_markers) > 0:
    sc.pl.dotplot(adata, var_names=all_markers, groupby='cell_type',
                  standard_scale='var', save='_07_dotplot_markers.png', show=False)
    src = os.path.join(FIGURES_TMP, "dotplot_07_dotplot_markers.png")
    if os.path.exists(src):
        shutil.move(src, os.path.join(OUTPUT_DIR, "07_dotplot_markers.png"))
        print(f"  保存: 07_dotplot_markers.png")

# [6.4] Heatmap
print("\n[6.4] 绘制 Heatmap ...")
if len(all_markers) > 0:
    sc.pl.heatmap(adata, var_names=all_markers, groupby='cell_type',
                  standard_scale='var', show_gene_labels=True,
                  save='_08_heatmap_markers.png', show=False)
    src = os.path.join(FIGURES_TMP, "heatmap_08_heatmap_markers.png")
    if os.path.exists(src):
        shutil.move(src, os.path.join(OUTPUT_DIR, "08_heatmap_markers.png"))
        print(f"  保存: 08_heatmap_markers.png")

# [6.5] 堆叠条形图
print("\n[6.5] 绘制聚类-细胞类型堆叠图 ...")
cluster_celltype = pd.crosstab(adata.obs['leiden_scvi'], adata.obs['cell_type'])
fig, ax = plt.subplots(figsize=(12, 5))
cluster_celltype.plot(kind='bar', stacked=True, colormap='tab20', ax=ax, width=0.85)
ax.set_xlabel('scVI Cluster'); ax.set_ylabel('Number of cells')
ax.set_title('Cell type composition per scVI cluster')
ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_cluster_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 09_cluster_composition.png")

# [6.6] LFC 小提琴图
print("\n[6.6] 绘制 LFC 分布 ...")
fig, ax = plt.subplots(figsize=(10, 6))
top_comparisons = de_results['comparison'].value_counts().head(6).index.tolist()
de_subset = de_results[de_results['comparison'].isin(top_comparisons)].copy()
if 'raw_normalized_mean1' in de_subset.columns:
    de_subset['lfc'] = np.log2(
        (de_subset['raw_normalized_mean1'].clip(lower=1e-10)) /
        (de_subset['raw_normalized_mean2'].clip(lower=1e-10))
    )
else:
    de_subset['lfc'] = np.log2(de_subset['scale1'] / de_subset['scale2'])
sns.violinplot(data=de_subset, x='comparison', y='lfc', ax=ax, palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Log2 Fold Change')
ax.set_title('LFC distribution across top comparisons')
ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No change')
ax.grid(alpha=0.3, axis='y')
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_de_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 10_de_violin.png")

# [6.7] 1v1 火山图
print("\n[6.7] 1v1 对比: Ventricular_Cardiomyocyte vs Fibroblast ...")
de_1v1 = model.differential_expression(
    groupby="cell_type",
    group1="Ventricular_Cardiomyocyte",
    group2="Fibroblast"
)
de_1v1.to_csv(os.path.join(OUTPUT_DIR, "de_1v1_VC_vs_FB.csv"))
fig, ax = plt.subplots(figsize=(8, 6))
if 'raw_normalized_mean1' in de_1v1.columns:
    de_1v1_lfc = np.log2(
        (de_1v1['raw_normalized_mean1'].clip(lower=1e-10)) /
        (de_1v1['raw_normalized_mean2'].clip(lower=1e-10))
    )
else:
    de_1v1_lfc = np.log2(de_1v1['scale1'] / de_1v1['scale2'])
de_1v1['log_p'] = -np.log10(de_1v1['proba_m1'].clip(lower=1e-300))
colors = np.where(
    (de_1v1_lfc > 1) & (de_1v1['proba_m1'] > 0.8), 'coral',
    np.where((de_1v1_lfc < -1) & (de_1v1['proba_m1'] > 0.8), 'steelblue', 'gray')
)
ax.scatter(de_1v1_lfc, de_1v1['log_p'], c=colors, s=8, alpha=0.6)
ax.set_xlabel('Log2 Fold Change (VC vs FB)')
ax.set_ylabel('-log10(P(DE))')
ax.set_title('Volcano: Ventricular Cardiomyocyte vs Fibroblast')
ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
ax.axvline(1, color='red', linestyle='--', alpha=0.5)
ax.axvline(-1, color='red', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_volcano_1v1.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 11_volcano_1v1.png")


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
