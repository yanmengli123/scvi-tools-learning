"""
Heart Atlas Complete scVI Analysis - Full Pipeline
6大模块：环境→预处理→scVI训练→提取结果→降维聚类→差异表达Marker
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
from matplotlib import rcParams

# ============================================================
# 模块1: 环境初始化
# ============================================================
print("=" * 70)
print("模块1: 环境初始化")
print("=" * 70)

np.random.seed(0)
torch.manual_seed(0)
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

sc.settings.verbosity = 1
sc.settings.set_figure_params(
    dpi=120, dpi_save=300,
    figsize=(6, 5), facecolor='white',
    vector_friendly=True, format='png'
)
sc.settings.figdir = './figures/'
os.makedirs(sc.settings.figdir, exist_ok=True)

use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
print(f"  Device: {device} | scvi-tools: {scvi.__version__}")

# ============================================================
# 模块2: 数据载入+预处理
# ============================================================
print("\n" + "=" * 70)
print("模块2: 数据载入+预处理")
print("=" * 70)

# 2.1 加载数据
print("\n[2.1] 加载原始数据...")
adata = sc.read_h5ad("heart_atlas.h5ad")
print(f"  原始数据: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

# 2.2 过滤低质量基因和细胞
print("\n[2.2] 质量过滤...")
# 过滤在少于3个细胞中表达的基因
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=100)
print(f"  过滤后: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

# 2.3 关键：原始 UMI 计数存入 layers["counts"]
print("\n[2.3] 保存原始 counts 到 layers...")
adata.layers["counts"] = adata.X.copy()
print(f"  layers['counts'] 形状: {adata.layers['counts'].shape}")

# 2.4 常规 log 归一化用于绘图
print("\n[2.4] log 归一化 (用于绘图)...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata  # 保存log-normalized 用于下游可视化

# 2.5 筛选高可变基因 HVG
print("\n[2.5] 高可变基因筛选 (HVG)...")
# n_top_genes 设为 1200
sc.pp.highly_variable_genes(
    adata, n_top_genes=1200,
    flavor='seurat', batch_key='donor',
    subset=True  # 立即子集化
)
print(f"  HVG 后: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

# 2.6 setup_anndata 告知 scVI
print("\n[2.6] setup_anndata ...")
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="donor",                      # 批次 (不同供体)
    categorical_covariate_keys=["cell_source"],  # 细胞来源
    layer="counts",                         # 关键：用原始 counts
    continuous_covariate_keys=["percent_mito", "percent_ribo"],  # 干扰因子
)
print("  setup_anndata 完成")

# 2.7 绘制预处理QC图
print("\n[2.7] 绘制预处理QC图...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# counts 分布
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
axes[0].hist(adata.obs['total_counts'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Total counts')
axes[0].set_ylabel('Number of cells')
axes[0].set_title('Total counts distribution')

# genes 分布
axes[1].hist(adata.obs['n_genes_by_counts'], bins=50, color='coral', edgecolor='white')
axes[1].set_xlabel('Genes per cell')
axes[1].set_ylabel('Number of cells')
axes[1].set_title('Genes per cell distribution')

# mito%
axes[2].hist(adata.obs['percent_mito'], bins=50, color='seagreen', edgecolor='white')
axes[2].set_xlabel('Percent mito')
axes[2].set_ylabel('Number of cells')
axes[2].set_title('Mito gene % distribution')

plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}01_preprocess_qc.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 01_preprocess_qc.png")

# ============================================================
# 模块3: SCVI 模型构建、训练、保存
# ============================================================
print("\n" + "=" * 70)
print("模块3: SCVI 模型构建与训练")
print("=" * 70)

# 3.1 构建模型 (ZINB 变分自编码器)
print("\n[3.1] 构建 SCVI 模型...")
model = scvi.model.SCVI(
    adata,
    n_latent=30,        # 隐空间维度
    n_layers=2,         # 编码器/解码器层数
    n_hidden=128,       # 隐藏层维度
    dropout_rate=0.1,
    dispersion="gene-batch",  # 每个基因每个批次色散
    gene_likelihood="nb",     # 负二项分布
)
print(f"  模型参数:")
print(f"    n_latent=30, n_layers=2, n_hidden=128")
print(f"    dispersion='gene-batch', gene_likelihood='nb'")

# 3.2 训练模型
print("\n[3.2] 训练模型 (自动 GPU 加速)...")
model.train(
    max_epochs=100,
    batch_size=128,
    train_size=0.9,
    early_stopping=True,
    enable_progress_bar=False,
    accelerator="auto",
)
print("  训练完成!")

# 3.3 绘制训练曲线
print("\n[3.3] 绘制训练曲线...")
train_history = model.history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

if 'train_loss_step' in train_history:
    axes[0].plot(train_history['train_loss_step'], label='train', color='steelblue')
if 'elbo_train' in train_history:
    axes[0].plot(train_history['elbo_train'], label='ELBO', color='coral')
if 'reconstruction_loss_train' in train_history:
    axes[0].plot(train_history['reconstruction_loss_train'], label='Recon Loss', color='seagreen')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)

if 'kl_train' in train_history:
    axes[1].plot(train_history['kl_train'], label='KL(train)', color='steelblue')
if 'kl_validation' in train_history:
    axes[1].plot(train_history['kl_validation'], label='KL(val)', color='coral')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('KL Divergence')
axes[1].set_title('KL Divergence')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}02_training_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 02_training_curve.png")

# 3.4 保存模型
print("\n[3.4] 保存模型...")
model.save("scvi_heart_model/", overwrite=True)
print(f"  模型已保存到: scvi_heart_model/")

# ============================================================
# 模块4: 提取模型结果
# ============================================================
print("\n" + "=" * 70)
print("模块4: 提取模型结果")
print("=" * 70)

# 4.1 隐空间表示
print("\n[4.1] 提取隐空间 X_scVI...")
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent
print(f"  X_scVI 形状: {latent.shape} (替代 PCA)")

# 4.2 归一化表达
print("\n[4.2] 提取归一化基因表达...")
normalized = model.get_normalized_expression(
    library_size=1e4, return_numpy=True
)
adata.layers["scvi_normalized"] = normalized
print(f"  scvi_normalized 形状: {normalized.shape}")

# 4.3 绘制隐空间分布
print("\n[4.3] 绘制隐空间分布...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(latent.flatten(), bins=100, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Latent value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Latent space values distribution')
axes[0].grid(alpha=0.3)

# Latent dim by variance
latent_var = np.var(latent, axis=0)
axes[1].bar(range(len(latent_var)), sorted(latent_var, reverse=True), color='coral')
axes[1].set_xlabel('Latent dimension (sorted)')
axes[1].set_ylabel('Variance')
axes[1].set_title('Variance explained by each latent dim')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}03_latent_space.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 03_latent_space.png")

# ============================================================
# 模块5: 降维可视化+聚类
# ============================================================
print("\n" + "=" * 70)
print("模块5: 降维可视化+聚类")
print("=" * 70)

# 5.1 对照组: 原始 PCA → UMAP
print("\n[5.1] 对照组: PCA + UMAP (批次效应可见)...")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca', key_added='pca')
sc.tl.umap(adata, neighbors_key='pca')
adata.obsm['X_umap_pca'] = adata.obsm['X_umap'].copy()
print("  PCA + UMAP 完成")

# 5.2 实验组: scVI 隐空间 → UMAP
print("\n[5.2] 实验组: X_scVI + UMAP (批次校正)...")
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_scVI', key_added='scvi')
sc.tl.umap(adata, neighbors_key='scvi')
adata.obsm['X_umap_scvi'] = adata.obsm['X_umap'].copy()
print("  scVI + UMAP 完成")

# 5.3 绘制批次校正对比图
print("\n[5.3] 绘制批次校正对比图 (PCA vs scVI)...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# PCA - donor
sc.pl.embedding(adata, basis='X_umap_pca', color='donor',
                ax=axes[0, 0], show=False, title='PCA UMAP - Donor (Batch)',
                frameon=False, legend_loc=None, size=8)
# PCA - cell_type
sc.pl.embedding(adata, basis='X_umap_pca', color='cell_type',
                ax=axes[0, 1], show=False, title='PCA UMAP - Cell Type',
                frameon=False, legend_loc='right margin', size=8, legend_fontsize=6)
# scVI - donor
sc.pl.embedding(adata, basis='X_umap_scvi', color='donor',
                ax=axes[1, 0], show=False, title='scVI UMAP - Donor (Batch)',
                frameon=False, legend_loc=None, size=8)
# scVI - cell_type
sc.pl.embedding(adata, basis='X_umap_scvi', color='cell_type',
                ax=axes[1, 1], show=False, title='scVI UMAP - Cell Type',
                frameon=False, legend_loc='right margin', size=8, legend_fontsize=6)

plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}04_batch_correction_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 04_batch_correction_comparison.png")

# 5.4 Leiden 聚类
print("\n[5.4] Leiden 聚类 (基于 scVI 隐空间)...")
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_scvi', neighbors_key='scvi')
n_clusters = adata.obs['leiden_scvi'].nunique()
print(f"  聚类数: {n_clusters}")
print(adata.obs['leiden_scvi'].value_counts().sort_index())

# 5.5 绘制聚类结果
print("\n[5.5] 绘制聚类结果...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sc.pl.embedding(adata, basis='X_umap_scvi', color='leiden_scvi',
                ax=axes[0], show=False, title=f'scVI Clusters (n={n_clusters})',
                frameon=False, legend_loc='on data', size=10, legend_fontsize=10)
sc.pl.embedding(adata, basis='X_umap_scvi', color='cell_type',
                ax=axes[1], show=False, title='Original Cell Types',
                frameon=False, legend_loc='right margin', size=10, legend_fontsize=8)

plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}05_clustering_result.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 05_clustering_result.png")

# 5.6 绘制聚类-细胞类型热图
print("\n[5.6] 绘制聚类-细胞类型对应热图...")
cluster_celltype = pd.crosstab(
    adata.obs['leiden_scvi'], adata.obs['cell_type']
)
# 标准化
cluster_celltype_norm = cluster_celltype.div(cluster_celltype.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(cluster_celltype_norm.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(cluster_celltype_norm.columns)))
ax.set_xticklabels(cluster_celltype_norm.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(cluster_celltype_norm.index)))
ax.set_yticklabels(cluster_celltype_norm.index)
ax.set_xlabel('Cell Type')
ax.set_ylabel('scVI Cluster')
ax.set_title('Cluster-CellType Correspondence (normalized)')

# 添加数值
for i in range(len(cluster_celltype_norm.index)):
    for j in range(len(cluster_celltype_norm.columns)):
        if cluster_celltype_norm.values[i, j] > 0.1:
            ax.text(j, i, f'{cluster_celltype_norm.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if cluster_celltype_norm.values[i, j] > 0.5 else 'black')

plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}06_cluster_celltype_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 06_cluster_celltype_heatmap.png")

# ============================================================
# 模块6: 贝叶斯差异表达+Marker绘图
# ============================================================
print("\n" + "=" * 70)
print("模块6: 贝叶斯差异表达+Marker绘图")
print("=" * 70)

# 6.1 一类vs全群 (1 vs all) - 找每种细胞类型marker
print("\n[6.1] 差异表达分析 (1 vs all 找每种细胞类型 marker)...")
de_results = model.differential_expression(
    groupby="cell_type",
)
print(f"  DE 结果形状: {de_results.shape}")
de_results.to_csv("differential_expression_all.csv")

# 6.2 选取每种细胞类型的 top markers
print("\n[6.2] 筛选每种细胞类型 top 5 marker genes...")
cell_types = adata.obs['cell_type'].unique().tolist()
top_markers_dict = {}

# 使用 raw_mean1/raw_mean2 来计算 log fold change
# 或使用 proba_m1 作为差异显著性的指标
for ct in cell_types:
    ct_de = de_results[de_results['comparison'] == f"{ct} vs Rest"]
    if len(ct_de) > 0:
        # 使用 raw_normalized_mean 计算 LFC (log2 fold change)
        if 'raw_normalized_mean1' in ct_de.columns and 'raw_normalized_mean2' in ct_de.columns:
            # 避免除零
            mean1 = ct_de['raw_normalized_mean1'].clip(lower=1e-10)
            mean2 = ct_de['raw_normalized_mean2'].clip(lower=1e-10)
            ct_de = ct_de.copy()
            ct_de['lfc'] = np.log2(mean1 / mean2)
            # 用 proba_m1 排序
            top_genes = ct_de.sort_values(['lfc', 'proba_m1'], ascending=[False, False]).head(5).index.tolist()
        else:
            top_genes = ct_de.sort_values('bayes_factor', ascending=False).head(5).index.tolist()
        top_markers_dict[ct] = top_genes
        print(f"  {ct}: {top_genes[:3]}...")

# 6.3 Dotplot 展示 marker 表达
print("\n[6.3] 绘制 Dotplot...")
# 选取前 10 种细胞类型
selected_cts = cell_types[:10] if len(cell_types) > 10 else cell_types
all_markers = []
for ct in selected_cts:
    if ct in top_markers_dict:
        all_markers.extend(top_markers_dict[ct][:3])
all_markers = list(dict.fromkeys(all_markers))[:20]  # 去重，最多 20

if len(all_markers) > 0:
    sc.pl.dotplot(
        adata, var_names=all_markers, groupby='cell_type',
        standard_scale='var',
        save=f'_07_dotplot_markers.png',
        show=False
    )
    # scanpy 默认保存到 figures/dotplot_*.png
    import shutil
    if os.path.exists("figures/dotplot_07_dotplot_markers.png"):
        shutil.move("figures/dotplot_07_dotplot_markers.png",
                    f"{sc.settings.figdir}07_dotplot_markers.png")
    print(f"  保存: 07_dotplot_markers.png")

# 6.4 Heatmap 展示 top markers
print("\n[6.4] 绘制 Heatmap...")
if len(all_markers) > 0:
    sc.pl.heatmap(
        adata, var_names=all_markers, groupby='cell_type',
        standard_scale='var', show_gene_labels=True,
        save=f'_08_heatmap_markers.png',
        show=False
    )
    if os.path.exists("figures/heatmap_08_heatmap_markers.png"):
        shutil.move("figures/heatmap_08_heatmap_markers.png",
                    f"{sc.settings.figdir}08_heatmap_markers.png")
    print(f"  保存: 08_heatmap_markers.png")

# 6.5 堆叠条形图：各聚类细胞类型组成
print("\n[6.5] 绘制聚类-细胞类型堆叠图...")
fig, ax = plt.subplots(figsize=(12, 5))
cluster_celltype.plot(kind='bar', stacked=True, colormap='tab20', ax=ax, width=0.85)
ax.set_xlabel('scVI Cluster')
ax.set_ylabel('Number of cells')
ax.set_title('Cell type composition per scVI cluster')
ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}09_cluster_composition.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 09_cluster_composition.png")

# 6.6 火山图: Volin-like visualization of top DE
print("\n[6.6] 绘制差异表达分布图...")
fig, ax = plt.subplots(figsize=(10, 6))
# 按 comparison 分组画 raw_normalized_mean1 的小提琴图
import seaborn as sns
top_comparisons = de_results['comparison'].value_counts().head(6).index.tolist()
de_subset = de_results[de_results['comparison'].isin(top_comparisons)].copy()
# 计算 log2 fold change
de_subset['lfc'] = np.log2(
    (de_subset['raw_normalized_mean1'].clip(lower=1e-10)) /
    (de_subset['raw_normalized_mean2'].clip(lower=1e-10))
)
sns.violinplot(data=de_subset, x='comparison', y='lfc', ax=ax, palette='Set2')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylabel('Log2 Fold Change')
ax.set_title('LFC distribution across top comparisons')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}10_de_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 10_de_violin.png")

# 6.7 两组对比 1v1 示例: Ventricular_Cardiomyocyte vs Fibroblast
print("\n[6.7] 1v1 对比: Ventricular_Cardiomyocyte vs Fibroblast...")
de_1v1 = model.differential_expression(
    groupby="cell_type",
    group1="Ventricular_Cardiomyocyte",
    group2="Fibroblast",
)
de_1v1.to_csv("de_1v1_VC_vs_FB.csv")

# 火山图
fig, ax = plt.subplots(figsize=(8, 6))
# 计算 LFC
de_1v1_lfc = np.log2(
    (de_1v1['raw_normalized_mean1'].clip(lower=1e-10)) /
    (de_1v1['raw_normalized_mean2'].clip(lower=1e-10))
)
de_1v1['log_p'] = -np.log10(de_1v1['proba_m1'].clip(lower=1e-300))
colors = np.where(
    (de_1v1_lfc > 1) & (de_1v1['proba_m1'] > 0.8), 'coral',
    np.where((de_1v1_lfc < -1) & (de_1v1['proba_m1'] > 0.8), 'steelblue', 'gray')
)
ax.scatter(de_1v1_lfc, de_1v1['log_p'], c=colors, s=8, alpha=0.6)
ax.set_xlabel('Log2 Fold Change')
ax.set_ylabel('-log10(probability DE)')
ax.set_title('Volcano: Ventricular Cardiomyocyte vs Fibroblast')
ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
ax.axvline(1, color='red', linestyle='--', alpha=0.5)
ax.axvline(-1, color='red', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{sc.settings.figdir}11_volcano_1v1.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: 11_volcano_1v1.png")

# ============================================================
# 最终保存
# ============================================================
print("\n" + "=" * 70)
print("最终保存")
print("=" * 70)

# 保存带分析的 AnnData
adata.write_h5ad("heart_atlas_analyzed.h5ad")
print(f"  AnnData 已保存: heart_atlas_analyzed.h5ad")

# 列出所有输出
print(f"\n生成的图片 (figures/):")
import glob
for f in sorted(glob.glob(f"{sc.settings.figdir}*.png")):
    size = os.path.getsize(f) / 1024
    print(f"  {os.path.basename(f)} ({size:.1f} KB)")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)

# 总结
print(f"\n=== 最终汇总 ===")
print(f"  总细胞数: {adata.n_obs}")
print(f"  总基因数(HVG): {adata.n_vars}")
print(f"  原始细胞类型: {adata.obs['cell_type'].nunique()}")
print(f"  scVI 聚类数: {adata.obs['leiden_scvi'].nunique()}")
print(f"  隐空间维度: {adata.obsm['X_scVI'].shape[1]}")
print(f"  差异表达基因对: {de_results.shape[0]}")
