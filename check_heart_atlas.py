"""
检查 heart_atlas.h5ad 数据内容
"""
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import anndata as ad

print("=" * 60)
print("Heart Atlas Dataset Inspection")
print("=" * 60)

# 加载数据
adata = sc.read_h5ad("heart_atlas.h5ad")

print(f"\n数据维度 (n_obs x n_vars): {adata.shape}")
print(f"细胞数: {adata.n_obs}")
print(f"基因数: {adata.n_vars}")

print("\n=== obs (细胞元信息) ===")
print(f"列名: {list(adata.obs.columns)}")
print(f"\n前5行:")
print(adata.obs.head())

print("\n=== var (基因元信息) ===")
print(f"列名: {list(adata.var.columns)}")
print(f"\n前5行:")
print(adata.var.head())

print("\n=== obsm (多维注释) ===")
print(f"键: {list(adata.obsm.keys())}")

print("\n=== uns (非结构化注释) ===")
print(f"键: {list(adata.uns.keys())}")

print("\n=== layers (额外数据层) ===")
print(f"键: {list(adata.layers.keys())}")

print("\n=== 数据类型 ===")
print(f"X type: {type(adata.X)}")
print(f"X dtype: {adata.X.dtype}")
print(f"X shape: {adata.X.shape}")

print("\n=== 数据统计 ===")
import numpy as np
if hasattr(adata.X, 'toarray'):
    X = adata.X.toarray()
else:
    X = adata.X
print(f"X min: {X.min():.3f}")
print(f"X max: {X.max():.3f}")
print(f"X mean: {X.mean():.3f}")
print(f"X std: {X.std():.3f}")
print(f"稀疏度: {(X == 0).sum() / X.size * 100:.2f}%")
