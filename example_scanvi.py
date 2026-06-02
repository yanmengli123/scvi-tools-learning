"""
scvi-tools SCANVI 示例
演示半监督细胞类型注释
"""

import scvi
import numpy as np
import scanpy as sc
from scvi.data import synthetic_iid

# 1. 生成合成数据
print("=== 生成合成数据 ===")
adata = synthetic_iid(
    batch_size=100,
    n_genes=50,
    n_batches=2,
    n_labels=3,
)

# 2. 模拟部分标签已知（半监督场景）
print("\n=== 设置半监督标签 ===")
# 假设只有一部分细胞有标签
np.random.seed(42)
n_cells = adata.n_obs
labeled_idx = np.random.choice(n_cells, size=int(n_cells * 0.3), replace=False)

# 创建标签列，未标注的设为 "Unknown"
adata.obs["cell_type"] = adata.obs["labels"].copy()
adata.obs.loc[~adata.obs.index.isin(adata.obs.index[labeled_idx]), "cell_type"] = "Unknown"

print(f"已标注细胞: {(adata.obs['cell_type'] != 'Unknown').sum()}")
print(f"未标注细胞: {(adata.obs['cell_type'] == 'Unknown').sum()}")

# 3. 设置 AnnData
print("\n=== 设置 AnnData ===")
scvi.model.SCANVI.setup_anndata(
    adata,
    batch_key="batch",
    labels_key="cell_type",
)

# 4. 创建 SCANVI 模型
print("\n=== 创建 SCANVI 模型 ===")
scanvae = scvi.model.SCANVI(
    adata,
    unlabeled_category="Unknown",  # 未标注类别
)

# 5. 训练模型
print("\n=== 训练模型 ===")
scanvae.train(
    max_epochs=50,
    train_size=0.8,
    early_stopping=True,
)

# 6. 预测未标注细胞的类型
print("\n=== 预测细胞类型 ===")
predictions = scanvae.predict()
print(f"预测结果维度: {predictions.shape}")
print(f"预测类别: {predictions.columns.tolist()}")

# 7. 获取预测的置信度
print("\n=== 获取置信度 ===")
probs = scanvae.predict(soft=True)
print(f"置信度维度: {probs.shape}")
print("\n示例预测概率:")
print(probs.head())

# 8. 差异表达分析
print("\n=== 差异表达分析 ===")
de_results = scanvae.differential_expression(
    groupby="cell_type",
    group1="label_0",
)
print(f"差异表达结果: {de_results.shape}")

print("\n=== 示例完成 ===")
