"""
Heart Atlas Full Analysis Pipeline
完整分析流程：数据加载 -> 预处理 -> 训练 -> 整合 -> 注释 -> 可视化
"""
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("Heart Atlas Full Analysis Pipeline")
print("=" * 60)

# ===========================
# 1. 加载数据
# ===========================
print("\n[1/8] Loading data...")
adata = sc.read_h5ad("heart_atlas.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
print(f"  Donors: {adata.obs['donor'].nunique()}")
print(f"  Cell sources: {adata.obs['cell_source'].unique().tolist()}")

# ===========================
# 2. 数据预处理
# ===========================
print("\n[2/8] Preprocessing...")
# 使用 counts 层
adata.layers["counts"] = adata.layers["counts"]
print(f"  Total counts: {adata.X.sum():.2e}")
print(f"  Cells with >0 counts: {(adata.obs['n_counts'] > 0).sum()}")

# ===========================
# 3. 设置 AnnData for scVI
# ===========================
print("\n[3/8] Setting up AnnData for scVI...")
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="donor",        # 批次效应来自不同供体
    categorical_covariate_keys=["cell_source"],  # 细胞来源
    layer="counts",           # 使用原始 counts
)

# ===========================
# 4. 训练 SCVI 模型
# ===========================
print("\n[4/8] Training SCVI model...")
model = scvi.model.SCVI(
    adata,
    n_latent=30,              # 潜在空间维度
    n_layers=2,               # 编码器/解码器层数
    n_hidden=128,             # 隐藏层维度
    dispersion="gene-batch",  # 色散参数
    gene_likelihood="nb",     # 负二项似然
)

# 训练（可调整 max_epochs）
model.train(
    max_epochs=100,
    batch_size=128,
    train_size=0.9,
    early_stopping=True,
    enable_progress_bar=False,
    accelerator="auto",  # 自动选择 GPU/CPU
)
print("  Training completed!")

# ===========================
# 5. 获取低维表示
# ===========================
print("\n[5/8] Extracting latent representation...")
# 潜在空间表示
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent
print(f"  Latent shape: {latent.shape}")

# 归一化表达
normalized = model.get_normalized_expression(
    library_size=1e4,
    return_numpy=True,
)
print(f"  Normalized expression shape: {normalized.shape}")

# ===========================
# 6. 降维与聚类
# ===========================
print("\n[6/8] Computing neighbors, UMAP, and clustering...")

# 使用 scVI 潜在表示计算邻居
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3)
sc.tl.leiden(adata, resolution=0.5, key_added="leiden_scvi")

print(f"  Clusters found: {adata.obs['leiden_scvi'].nunique()}")
print(f"  Cluster distribution:")
print(adata.obs['leiden_scvi'].value_counts().sort_index())

# ===========================
# 7. 差异表达分析
# ===========================
print("\n[7/8] Differential expression analysis...")
# 与原始 cell_type 对比
de_results = model.differential_expression(
    groupby="cell_type",
)
print(f"  DE results shape: {de_results.shape}")

# 获取每个聚类的 marker genes
print("\n  Top 5 marker genes per cluster:")
top_markers = {}
for cluster in sorted(adata.obs['leiden_scvi'].unique()):
    cluster_cells = adata.obs['leiden_scvi'] == cluster
    # 获取该聚类对应的细胞类型
    cell_types_in_cluster = adata.obs.loc[cluster_cells, 'cell_type'].value_counts()
    if len(cell_types_in_cluster) > 0:
        top_markers[cluster] = cell_types_in_cluster.head(3).to_dict()
print(top_markers)

# ===========================
# 8. 可视化
# ===========================
print("\n[8/8] Generating visualizations...")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# UMAP 按 cell_type 着色
sc.pl.umap(
    adata, color="cell_type",
    ax=axes[0, 0], show=False,
    title="Cell Types", frameon=False,
    legend_loc='on data',
    legend_fontsize=6,
)

# UMAP 按 cluster 着色
sc.pl.umap(
    adata, color="leiden_scvi",
    ax=axes[0, 1], show=False,
    title="scVI Clusters", frameon=False,
    legend_loc='on data',
    legend_fontsize=8,
)

# UMAP 按 donor 着色
sc.pl.umap(
    adata, color="donor",
    ax=axes[1, 0], show=False,
    title="Donor (Batch)", frameon=False,
    legend_loc='right margin',
    legend_fontsize=6,
)

# UMAP 按 cell_source 着色
sc.pl.umap(
    adata, color="cell_source",
    ax=axes[1, 1], show=False,
    title="Cell Source", frameon=False,
    legend_loc='right margin',
    legend_fontsize=8,
)

plt.tight_layout()
plt.savefig("heart_atlas_analysis.png", dpi=150, bbox_inches="tight")
print("  Saved: heart_atlas_analysis.png")

# ===========================
# 保存结果
# ===========================
print("\n[Save] Saving results...")

# 保存模型
model.save("scvi_heart_atlas_model/", overwrite=True)
print("  Model saved: scvi_heart_atlas_model/")

# 保存带分析的 AnnData
output_path = "heart_atlas_analyzed.h5ad"
adata.write_h5ad(output_path)
print(f"  AnnData saved: {output_path}")

# 保存差异表达结果
de_results.to_csv("differential_expression.csv")
print("  DE results saved: differential_expression.csv")

# 保存聚类信息
cluster_summary = pd.DataFrame({
    'cluster': adata.obs['leiden_scvi'],
    'cell_type': adata.obs['cell_type'],
    'donor': adata.obs['donor'],
    'cell_source': adata.obs['cell_source'],
})
cluster_summary.to_csv("cluster_summary.csv", index=False)
print("  Cluster summary saved: cluster_summary.csv")

print("\n" + "=" * 60)
print("Analysis completed successfully!")
print("=" * 60)

# 打印总结
print("\n=== Summary ===")
print(f"Total cells: {adata.n_obs}")
print(f"Total genes: {adata.n_vars}")
print(f"Cell types: {adata.obs['cell_type'].nunique()}")
print(f"scVI clusters: {adata.obs['leiden_scvi'].nunique()}")
print(f"\nTop cell types:")
print(adata.obs['cell_type'].value_counts().head(10))
