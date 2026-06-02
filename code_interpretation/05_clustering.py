"""
================================================================================
模块5：降维可视化 + 细胞聚类 - scVI 的核心应用
================================================================================
【本模块三大功能】
    1. 对照组：PCA → UMAP（看批次效应）
    2. 实验组：X_scVI → UMAP（看批次校正效果）
    3. Leiden 聚类（基于 X_scVI 的图聚类）

【UMAP 是什么？】
    Uniform Manifold Approximation and Projection
    - 非线性降维算法
    - 输入：高维数据的图结构（k-NN graph）
    - 输出：2D 坐标
    - 保持局部结构 + 全局拓扑
    - 比 t-SNE 保留更多全局信息

【Leiden 聚类算法】
    - 基于图的社区检测算法
    - 输入：k-NN graph（sc.pp.neighbors 已经建好）
    - 输出：每个细胞的 cluster 标签
    - 优点：保证社区内连接稠密，社区间稀疏
    - resolution 参数：越大聚类越多

【resolution 选择技巧】
    0.3 - 0.5  → 粗粒度（大类）
    0.5 - 1.0  → 中等（推荐）
    1.0 - 2.0  → 细粒度（亚型）
    > 2.0      → 过度聚类（每簇细胞太少）
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


# ============================================================
# 步骤 5.1：加载数据
# ============================================================
print("=" * 70)
print("模块5：降维可视化 + 聚类")
print("=" * 70)

adata = sc.read_h5ad(os.path.join(OUTPUT_DIR, "with_scvi_adata.h5ad"))
# 这个 h5ad 已经包含：
#   - X_scVI (30 维隐空间)
#   - scvi_normalized (去噪后的表达)
#   - 原始 obs/var 信息
print(f"  数据加载: {adata.shape}")


# ============================================================
# 步骤 5.2：对照组 - PCA + UMAP (会看到批次效应)
# ============================================================
print("\n[5.1] 对照组: PCA + UMAP (批次效应可见)...")
# PCA 是线性降维，不考虑批次
# 我们用它来对比：scVI 到底去除了多少批次效应

sc.pp.scale(adata, max_value=10)
# scale: z-score 标准化 (x - mean) / std
# max_value=10 防止极值影响
# PCA 对量级敏感，所以必须先 scale

sc.tl.pca(adata, n_comps=30)
# tl.pca 计算前 30 个主成分
# 默认存到 adata.obsm['X_pca'] 和 adata.varm['PCs']

sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca', key_added='pca')
# pp.neighbors 构建 k-NN 图
# use_rep='X_pca' 告诉它用 X_pca 来算距离
# key_added='pca' 防止覆盖其他邻居图

sc.tl.umap(adata, neighbors_key='pca')
# tl.umap 基于 k-NN 图算 UMAP 坐标
# 默认存到 adata.obsm['X_umap']

adata.obsm['X_umap_pca'] = adata.obsm['X_umap'].copy()
# 备份一份到 X_umap_pca，方便后续对比
print(f"  PCA + UMAP 完成")


# ============================================================
# 步骤 5.3：实验组 - scVI + UMAP (批次校正)
# ============================================================
print("\n[5.2] 实验组: X_scVI + UMAP (批次校正)...")
# X_scVI 已经去除了批次（scVI 在训练时学到的）
# 所以 UMAP 应该看到：
#   - 不同 donor 颜色混合
#   - 不同 cell_type 明显分离

sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_scVI', key_added='scvi')
# 用 X_scVI 算邻居（替代 X_pca）
# 这里不调 scale，因为 X_scVI 已经是高斯分布了

sc.tl.umap(adata, neighbors_key='scvi')
# 算 scVI-UMAP

adata.obsm['X_umap_scvi'] = adata.obsm['X_umap'].copy()
# 备份
print(f"  scVI + UMAP 完成")


# ============================================================
# 步骤 5.4：绘制批次校正对比图 ★最重要的图★
# ============================================================
print("\n[5.3] 绘制批次校正对比图 (PCA vs scVI)...")
# 4 个子图：
#   左上：PCA UMAP - donor (批次)
#   右上：PCA UMAP - cell_type
#   左下：scVI UMAP - donor (批次)   ← 应该混合
#   右下：scVI UMAP - cell_type      ← 应该仍然分离

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 第一行：PCA (对照组)
sc.pl.embedding(adata, basis='X_umap_pca', color='donor',
                ax=axes[0, 0], show=False,
                title='PCA UMAP - Donor (Batch effect visible)',
                frameon=False, size=8, legend_loc=None)

sc.pl.embedding(adata, basis='X_umap_pca', color='cell_type',
                ax=axes[0, 1], show=False,
                title='PCA UMAP - Cell Type',
                frameon=False, size=8, legend_loc='right margin',
                legend_fontsize=6)

# 第二行：scVI (实验组)
sc.pl.embedding(adata, basis='X_umap_scvi', color='donor',
                ax=axes[1, 0], show=False,
                title='scVI UMAP - Donor (Batch effect removed)',
                frameon=False, size=8, legend_loc=None)

sc.pl.embedding(adata, basis='X_umap_scvi', color='cell_type',
                ax=axes[1, 1], show=False,
                title='scVI UMAP - Cell Type',
                frameon=False, size=8, legend_loc='right margin',
                legend_fontsize=6)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_batch_correction_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/04_batch_correction_comparison.png")


# ============================================================
# 步骤 5.5：Leiden 聚类
# ============================================================
print("\n[5.4] Leiden 聚类 (基于 scVI 隐空间)...")
# Leiden 是无监督聚类（不需要 cell_type 标签）
# 它的目标是让聚类结果与真实生物学一致
# 评估：聚类 vs 真实 cell_type 的重叠程度

sc.tl.leiden(
    adata,
    resolution=0.5,                    # 越大聚类越多
    key_added='leiden_scvi',          # 聚类结果存到 obs['leiden_scvi']
    neighbors_key='scvi'              # 用 scVI 邻居图
)
# 算法：迭代合并社区，每次合并后只接受质量提升的合并
# 相比 Louvain 更稳定，能保证社区内部全连接

n_clusters = adata.obs['leiden_scvi'].nunique()
print(f"  聚类数: {n_clusters}")
print(f"  聚类分布:")
print(adata.obs['leiden_scvi'].value_counts().sort_index())


# ============================================================
# 步骤 5.6：绘制聚类结果
# ============================================================
print("\n[5.5] 绘制聚类结果 ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：scVI 聚类（无监督结果）
sc.pl.embedding(adata, basis='X_umap_scvi', color='leiden_scvi',
                ax=axes[0], show=False,
                title=f'scVI Clusters (n={n_clusters}, resolution=0.5)',
                frameon=False, legend_loc='on data',
                size=12, legend_fontsize=10)

# 右图：真实细胞类型（监督标签）
sc.pl.embedding(adata, basis='X_umap_scvi', color='cell_type',
                ax=axes[1], show=False,
                title='Original Cell Types (Ground Truth)',
                frameon=False, legend_loc='right margin',
                size=12, legend_fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_clustering_result.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/05_clustering_result.png")


# ============================================================
# 步骤 5.7：聚类 - 细胞类型对应热图
# ============================================================
print("\n[5.6] 绘制聚类-细胞类型对应热图 ...")
# 交叉表：每个聚类中各细胞类型的数量
cluster_celltype = pd.crosstab(
    adata.obs['leiden_scvi'],     # 行：scVI 聚类
    adata.obs['cell_type']        # 列：真实细胞类型
)

# 行归一化（每个聚类中各细胞类型占比）
cluster_celltype_norm = cluster_celltype.div(cluster_celltype.sum(axis=1), axis=0)
# 这样能直接看出"聚类 X 主要包含细胞类型 Y"

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(cluster_celltype_norm.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(cluster_celltype_norm.columns)))
ax.set_xticklabels(cluster_celltype_norm.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(cluster_celltype_norm.index)))
ax.set_yticklabels(cluster_celltype_norm.index)
ax.set_xlabel('Cell Type', fontsize=12)
ax.set_ylabel('scVI Cluster', fontsize=12)
ax.set_title('Cluster-CellType Correspondence (row-normalized)', fontsize=13)

# 在每格写上比例
for i in range(len(cluster_celltype_norm.index)):
    for j in range(len(cluster_celltype_norm.columns)):
        if cluster_celltype_norm.values[i, j] > 0.1:  # 只显示 >10% 的
            ax.text(j, i, f'{cluster_celltype_norm.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if cluster_celltype_norm.values[i, j] > 0.5 else 'black')

plt.colorbar(im, ax=ax, label='Proportion')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_cluster_celltype_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/06_cluster_celltype_heatmap.png")
print(f"  解读:")
print(f"    - 颜色深 → 该聚类中该细胞类型占比高")
print(f"    - 对角线明显 → 聚类与细胞类型一致（理想）")

# 保存
adata.write_h5ad(os.path.join(OUTPUT_DIR, "clustered_adata.h5ad"))
print(f"\n  已保存: clustered_adata.h5ad")
