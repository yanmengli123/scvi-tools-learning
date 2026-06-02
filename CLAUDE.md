# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一个 **scvi-tools 学习项目**：使用真实的 scvi-tools 库（PyTorch + AnnData）分析人心脏单细胞图谱数据集 `heart_atlas.h5ad`。项目内不修改 scvi-tools 源代码，只调用其 API 完成分析。

源 scvi-tools 库代码在 `src/scvi/`（v1.4.3，仅作阅读参考）。学习者**主要编写/编辑的是根目录的 Python 脚本和 `code_interpretation/`**。

## 数据集

- `heart_atlas.h5ad`（76MB，未提交，需自行提供）— 人心脏单细胞图谱
  - 18,641 细胞 × 1,200 高变基因
  - 关键 obs 列：`cell_type`（11种）、`donor`（14个，批次键）、`cell_source`（4种）、`region`（6个：LV/AX/RV/SP/LA/RA）、`percent_mito`、`percent_ribo`
  - 已含 `X_scVI` 隐空间、`X_umap` 坐标、`layers['counts']`（原始 UMI 计数）
  - **无 spatial 坐标**（不支持真正的空间映射）

## Python 环境

- 指定解释器：`D:\soft\Python310\python.exe`
- 关键依赖：scvi-tools 1.3.3、scanpy、anndata、pytorch、lightning、flax、numpyro
- ⚠️ numpyro 与 jax 版本有兼容性问题，必须装 **`numpyro<0.20`**（当前 jax 0.6.2）
- ⚠️ Windows 长路径问题导致部分包需用 `--no-deps` 安装（flax/optax/numpyro）

## 运行命令

### 根目录示例脚本（合成数据，无需 h5ad）

```bash
"D:\soft\Python310\python.exe" example_basic.py        # SCVI 基础
"D:\soft\Python310\python.exe" example_scanvi.py       # SCANVI 半监督
"D:\soft\Python310\python.exe" example_integration.py  # 批次整合
"D:\soft\Python310\python.exe" example_demo.py         # 简洁输出版
```

### heart_atlas 实战分析

```bash
"D:\soft\Python310\python.exe" check_heart_atlas.py          # 数据探查
"D:\soft\Python310\python.exe" pipeline_full_analysis.py     # 6 模块完整流程
"D:\soft\Python310\python.exe" heart_atlas_full_analysis.py  # 同上早期版本
```

### 源码解读（带详细中文注释）

```bash
cd code_interpretation
"D:\soft\Python310\python.exe" pipeline_e2e.py  # 推荐：端到端一键跑通
# 或分模块:
"D:\soft\Python310\python.exe" 01_env_init.py
"D:\soft\Python310\python.exe" 02_data_preprocess.py
# 注意：分模块运行需在内存中保留 adata，建议直接用 pipeline_e2e.py
```

## 6 大模块分析流程

scVI 实战的标准流水线（pipeline_full_analysis.py 和 pipeline_e2e.py 都按此结构）：

| 模块 | 内容 | 关键函数 |
|------|------|----------|
| 1. 环境初始化 | 固定 seed、配置 Scanpy、检测 GPU | `np.random.seed(0)`, `sc.settings.set_figure_params` |
| 2. 数据预处理 | 过滤、**保存原始 count 到 layers**、HVG、setup_anndata | `scvi.model.SCVI.setup_anndata(layer="counts", batch_key="donor")` |
| 3. SCVI 训练 | 构建 VAE、ZINB 似然、训练 100 epoch | `scvi.model.SCVI(n_latent=30, gene_likelihood="nb")` |
| 4. 提取结果 | X_scVI 隐空间、归一化表达 | `model.get_latent_representation()`, `model.get_normalized_expression()` |
| 5. 降维聚类 | PCA 对照组 vs scVI 实验组、Leiden 聚类 | `sc.pp.neighbors(use_rep='X_scVI')`, `sc.tl.leiden()` |
| 6. 差异表达 | 贝叶斯 DE 1vall/1v1、Dotplot/Heatmap/火山图 | `model.differential_expression(groupby="cell_type")` |

## 关键坑点（必读）

1. **scVI 必须用原始 UMI count**：必须 `adata.layers["counts"] = adata.X.copy()` 后用 `setup_anndata(layer="counts", ...)`。log 归一化数据会让模型失效。

2. **scVI 注册信息不持久化到 h5ad**：分模块保存 h5ad 后再 load 会报错 "Please set up your AnnData with SCVI.setup_anndata first"。**解决方案：使用 `pipeline_e2e.py` 端到端运行，或在每个新脚本里重做 `setup_anndata`**。

3. **scvi-tools 1.3.3 的 DE 结果列名**：`lfc_mean` 列不存在！需用 `raw_normalized_mean1`/`raw_normalized_mean2` 自己算 `lfc = log2(m1/m2)`，差异显著性用 `proba_m1`。

4. **scanpy 保存图片路径**：默认会保存到 `figures/`，导致文件名前缀叠加（`figuresfigures/...`）。要明确设置 `sc.settings.figdir`。

5. **代码位置 ≠ 数据位置**：`code_interpretation/` 下的脚本使用相对路径 `../heart_atlas.h5ad`，所有产物输出到 `code_interpretation_output/`。

## 输出位置

- 根目录脚本 → 散落在根目录（`figures01_*.png` 等）
- `code_interpretation/pipeline_e2e.py` → `code_interpretation_output/`
- 大文件（h5ad、模型权重）已加入 `.gitignore`，**不会被提交**

## 仓库约定

- 不要提交 `*.h5ad`、`*.png`、模型目录
- 提交前 `git status` 检查工作树
- 主分支：`main`，远程：`https://github.com/yanmengli123/scvi-tools-learning.git`
