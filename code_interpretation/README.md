# scVI 实战代码源码解读

> 配套数据集: `heart_atlas.h5ad` (人类心脏单细胞图谱, 18,641 细胞 × 1,200 基因)

## 📁 文件清单

### 分模块源码（6 个文件，每个独立可读）

| # | 文件 | 功能 | 关键技术 |
|---|------|------|----------|
| 1 | [01_env_init.py](01_env_init.py) | 环境初始化 | 随机种子、GPU 检测、Scanpy 配置 |
| 2 | [02_data_preprocess.py](02_data_preprocess.py) | 数据加载与预处理 | 原始 count 保存、HVG 筛选、setup_anndata |
| 3 | [03_scvi_train.py](03_scvi_train.py) | scVI 模型训练 | VAE 架构、ZINB 分布、训练曲线 |
| 4 | [04_extract_results.py](04_extract_results.py) | 提取模型结果 | X_scVI 隐空间、归一化表达 |
| 5 | [05_clustering.py](05_clustering.py) | 降维可视化与聚类 | UMAP、Leiden 聚类 |
| 6 | [06_de_markers.py](06_de_markers.py) | 差异表达与 Marker | 贝叶斯 DE、Dotplot/Heatmap、火山图 |

### 一键运行脚本

- [pipeline_e2e.py](pipeline_e2e.py) - 端到端流程（2-6 模块整合版）

## 🚀 推荐运行方式（两种）

### 方式 A: 端到端（最简单）

```bash
cd code_interpretation
"D:\soft\Python310\python.exe" pipeline_e2e.py
```

### 方式 B: 分模块运行（学习用）

```bash
cd code_interpretation
"D:\soft\Python310\python.exe" 01_env_init.py
"D:\soft\Python310\python.exe" 02_data_preprocess.py  → 生成 processed_adata.h5ad
"D:\soft\Python310\python.exe" 03_scvi_train.py       → 生成 scvi_heart_model/
"D:\soft\Python310\python.exe" 04_extract_results.py  → 生成 with_scvi_adata.h5ad
"D:\soft\Python310\python.exe" 05_clustering.py       → 生成 clustered_adata.h5ad
"D:\soft\Python310\python.exe" 06_de_markers.py       → 全部完成
```

> **注意**: 分模块运行需要 h5ad 文件中保留 scVI 注册信息。如果遇到 `Please set up your AnnData with SCVI.setup_anndata first` 错误，请改用 `pipeline_e2e.py`。

## 📂 输出位置

所有产物在 `code_interpretation_output/`:

```
code_interpretation_output/
├── 01_preprocess_qc.png          ← 11 张可视化图片
├── 02_training_curve.png
├── 03_latent_space.png
├── 04_batch_correction_comparison.png  (★最重要：PCA vs scVI 批次校正对比)
├── 05_clustering_result.png
├── 06_cluster_celltype_heatmap.png
├── 07_dotplot_markers.png        (在 scanpy_tmp/，需手动 mv)
├── 08_heatmap_markers.png
├── 09_cluster_composition.png
├── 10_de_violin.png
├── 11_volcano_1v1.png
├── differential_expression_all.csv
├── de_1v1_VC_vs_FB.csv
├── processed_adata.h5ad
├── scvi_heart_model/             (模型权重)
└── _scanpy_tmp/                  (scanpy 临时输出)
```

## 📚 核心概念速查

| 概念 | 含义 | 在哪个文件出现 |
|------|------|---------------|
| **原始 count** | 整数计数，scVI 必须输入 | 模块2 步骤 2.3 |
| **ZINB** | 零膨胀负二项分布 | 模块3 步骤 3.1 |
| **隐空间 X_scVI** | 30 维，去批次后的表示 | 模块4 步骤 4.1 |
| **UMAP** | 非线性降维 | 模块5 步骤 5.2-5.3 |
| **Leiden** | 图聚类算法 | 模块5 步骤 5.4 |
| **proba_de** | 差异表达后验概率 | 模块6 步骤 6.2 |

## 🔧 关键代码片段

### 1. 原始 count 必须保存到 layers

```python
adata.layers["counts"] = adata.X.copy()
scvi.model.SCVI.setup_anndata(adata, layer="counts", ...)
```

### 2. setup_anndata 关键参数

```python
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="donor",                  # 批次（不同供体）
    categorical_covariate_keys=["cell_source"],
    layer="counts",                     # 原始 count 路径
    continuous_covariate_keys=["percent_mito", "percent_ribo"],
)
```

### 3. SCVI 模型

```python
model = scvi.model.SCVI(
    adata, n_latent=30, n_layers=2, n_hidden=128,
    dispersion="gene-batch", gene_likelihood="nb"
)
```

### 4. 提取结果

```python
latent = model.get_latent_representation()  # X_scVI
normalized = model.get_normalized_expression()  # 去噪表达
```

### 5. 差异表达

```python
# 1vall: 找每种细胞类型 marker
de = model.differential_expression(groupby="cell_type")

# 1v1: 两组对比
de = model.differential_expression(
    groupby="cell_type",
    group1="Ventricular_Cardiomyocyte",
    group2="Fibroblast"
)
```

## 🎯 阅读建议

1. 先看 [01_env_init.py](01_env_init.py) 了解环境配置
2. 重点看 [02_data_preprocess.py](02_data_preprocess.py) 的 `setup_anndata` 部分（最关键）
3. 重点看 [03_scvi_train.py](03_scvi_train.py) 的超参数说明
4. 重点看 [05_clustering.py](05_clustering.py) 的批次校正对比
5. 重点看 [06_de_markers.py](06_de_markers.py) 的 DE 结果解读
