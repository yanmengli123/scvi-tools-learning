"""
scvi-tools 快速入门示例
使用合成数据演示 SCVI 模型的基本用法
"""

import scvi
import numpy as np
import scanpy as sc
from scvi.data import synthetic_iid

# 1. 生成合成数据（不需要下载）
print("=== 生成合成数据 ===")
adata = synthetic_iid(
    batch_size=100,  # 每个批次的细胞数
    n_genes=50,      # 基因数
    n_batches=2,     # 批次数
    n_labels=3,      # 细胞类型数
)
print(f"数据维度: {adata.shape}")
print(f"批次信息: {adata.obs['batch'].unique()}")
print(f"细胞类型: {adata.obs['labels'].unique()}")

# 2. 设置 AnnData
print("\n=== 设置 AnnData ===")
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="batch",  # 批次信息
    labels_key="labels" # 细胞类型标签
)

# 3. 创建并训练模型
print("\n=== 训练 SCVI 模型 ===")
model = scvi.model.SCVI(adata)
model.train(
    max_epochs=50,
    train_size=0.8,
    early_stopping=True,
)

# 4. 获取低维表示（latent representation）
print("\n=== 获取低维表示 ===")
latent = model.get_latent_representation()
print(f"潜在空间维度: {latent.shape}")

# 5. 获取重构的基因表达
print("\n=== 获取重构表达 ===")
denoised = model.get_normalized_expression()
print(f"重构表达维度: {denoised.shape}")

# 6. 获取差异表达基因
print("\n=== 差异表达分析 ===")
de_results = model.differential_expression(
    groupby="labels",
    group1="label_0",
    group2="label_1",
)
print(f"差异表达结果: {de_results.shape}")
print("\nTop 5 上调基因:")
print(de_results.head())

# 7. 保存和加载模型
print("\n=== 保存模型 ===")
model.save("scvi_model/", overwrite=True)
print("模型已保存到 scvi_model/")

print("\n=== 示例完成 ===")
