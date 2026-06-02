"""
================================================================================
模块4：提取模型结果 - 拿到 scVI 学习的"成果"
================================================================================
【核心目的】
    scVI 训练完成后能输出两类关键结果：
    1. 隐空间表示 X_scVI (替代 PCA) → 用于 UMAP/聚类
    2. 归一化基因表达 → 用于差异表达、画 marker

【X_scVI vs PCA 的区别】
    PCA 是线性降维（前 30 个主成分）
    scVI 是非线性深度学习降维
    X_scVI 比 X_pca 强在：
      - 自动去除批次效应
      - 保留非线性结构
      - 重建误差可作为质控指标

【get_normalized_expression 的作用】
    输入：原始 count + 协变量
    输出：scVI "想象"这个细胞在没有批次/技术噪声下的真实表达
    用途：
      - 找差异基因更准（去除技术差异）
      - 画 marker 表达更清晰
      - 替代 log 归一化
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
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
# 步骤 4.1：加载数据和模型
# ============================================================
print("=" * 70)
print("模块4：提取模型结果")
print("=" * 70)

adata = sc.read_h5ad(os.path.join(OUTPUT_DIR, "processed_adata.h5ad"))
model = scvi.model.SCVI.load(
    os.path.join(OUTPUT_DIR, "scvi_heart_model") + "/",
    adata=adata
)
print(f"  数据加载: {adata.shape}")
print(f"  模型加载: {type(model).__name__}")


# ============================================================
# 步骤 4.2：提取隐空间表示 X_scVI
# ============================================================
print("\n[4.1] 提取隐空间 X_scVI ...")
# get_latent_representation 内部流程：
#   1. 把所有细胞喂给编码器
#   2. 编码器输出 μ（30维）
#   3. μ 就是隐空间表示
#   4. 不取随机采样 (μ + σ*ε)，直接用 μ 更稳定

latent = model.get_latent_representation()
# 返回 (n_cells, 30) 的 numpy array

adata.obsm["X_scVI"] = latent
# 存入 AnnData，方便后续 scanpy 操作
# adata.obsm 是存"二维数组"的地方（X_scVI, X_umap, X_pca 等）

print(f"  X_scVI 形状: {latent.shape}")
print(f"  X_scVI 范围: [{latent.min():.2f}, {latent.max():.2f}]")
print(f"  X_scVI 均值={latent.mean():.2f}, 标准差={latent.std():.2f}")


# ============================================================
# 步骤 4.3：提取归一化基因表达
# ============================================================
print("\n[4.2] 提取归一化基因表达 ...")
# get_normalized_expression 内部流程：
#   1. 编码器输出 z
#   2. 解码器用 z 重建表达 (mean, dispersion, dropout)
#   3. 从 ZINB 分布中采样/求期望
#   4. library_size 缩放 (这里设为 1e4)
#   5. 考虑批次/协变量

normalized = model.get_normalized_expression(
    library_size=1e4,             # 把每个细胞归一化到 1e4 total counts
    return_numpy=True             # 返回 numpy array 而不是 dataframe
)
# 形状: (n_cells, n_genes)

adata.layers["scvi_normalized"] = normalized
# 存入 layers，方便后续使用
# 之后画 marker 基因用 adata.layers["scvi_normalized"]

print(f"  scvi_normalized 形状: {normalized.shape}")
print(f"  scvi_normalized 范围: [{normalized.min():.3f}, {normalized.max():.3f}]")


# ============================================================
# 步骤 4.4：绘制隐空间分布
# ============================================================
print("\n[4.3] 绘制隐空间分布 ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：所有隐空间值的分布（应该是接近高斯）
axes[0].hist(latent.flatten(), bins=100, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Latent value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Latent space values distribution\n(should look like Gaussian N(0,1))', fontsize=13)
axes[0].grid(alpha=0.3)
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='zero')
axes[0].legend()

# 右图：每个 latent dim 的方差（按降序）
latent_var = np.var(latent, axis=0)
sorted_idx = np.argsort(latent_var)[::-1]
axes[1].bar(range(len(latent_var)), latent_var[sorted_idx], color='coral', alpha=0.8)
axes[1].set_xlabel('Latent dimension (sorted by variance)', fontsize=12)
axes[1].set_ylabel('Variance', fontsize=12)
axes[1].set_title('Variance of each latent dimension', fontsize=13)
axes[1].grid(alpha=0.3)
# 如果很多 dim 方差都很大 → 编码器没压缩好
# 如果只有少数 dim 方差大 → 编码器找到了有效表示

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_latent_space.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  保存: {OUTPUT_DIR}/03_latent_space.png")

# 保存结果
adata.write_h5ad(os.path.join(OUTPUT_DIR, "with_scvi_adata.h5ad"))
print(f"\n  AnnData 已保存: with_scvi_adata.h5ad")
print(f"  - X_scVI: adata.obsm['X_scVI'] 形状 {adata.obsm['X_scVI'].shape}")
print(f"  - scvi_normalized: adata.layers['scvi_normalized'] 形状 {adata.layers['scvi_normalized'].shape}")
