"""
scvi-tools 数据整合示例
演示如何使用 SCVI 整合多个批次的数据
"""

import scvi
import numpy as np
import scanpy as sc
from scvi.data import synthetic_iid

# 1. 生成带有批次效应的合成数据
print("=== 生成多批次数据 ===")
adata = synthetic_iid(
    batch_size=200,  # 每个批次 200 个细胞
    n_genes=100,     # 100 个基因
    n_batches=3,     # 3 个批次
    n_labels=5,      # 5 种细胞类型
)
print(f"总细胞数: {adata.n_obs}")
print(f"批次数: {adata.obs['batch'].nunique()}")

# 2. 设置 AnnData
print("\n=== 设置 AnnData ===")
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="batch",
)

# 3. 训练 SCVI 模型（用于整合）
print("\n=== 训练整合模型 ===")
model = scvi.model.SCVI(adata)
model.train(
    max_epochs=100,
    train_size=0.8,
    early_stopping=True,
)

# 4. 获取整合后的低维表示
print("\n=== 获取整合表示 ===")
latent = model.get_latent_representation()
print(f"整合后维度: {latent.shape}")

# 5. 添加到 AnnData 用于可视化
adata.obsm["X_scVI"] = latent

# 6. 使用整合表示进行聚类
print("\n=== 基于整合表示聚类 ===")
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata, resolution=0.5)
sc.tl.umap(adata)

print(f"聚类数: {adata.obs['leiden'].nunique()}")

# 7. 保存整合后的数据
print("\n=== 保存结果 ===")
adata.write("integrated_data.h5ad")
print("整合数据已保存到 integrated_data.h5ad")

# 8. 获取批次校正后的表达（可选）
print("\n=== 获取校正后的表达 ===")
# 这可以用于下游分析
corrected_expr = model.get_normalized_expression()
print(f"校正表达维度: {corrected_expr.shape}")

print("\n=== 示例完成 ===")
print("\n你可以使用 scanpy 可视化整合结果:")
print("  sc.pl.umap(adata, color=['batch', 'labels', 'leiden'])")
