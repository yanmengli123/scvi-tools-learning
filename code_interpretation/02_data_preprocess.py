"""
================================================================================
模块2：数据载入与预处理 - scVI 分析的基础
================================================================================
【核心目的】
    1. 加载 heart_atlas.h5ad（人类心脏单细胞图谱）
    2. 质量控制（QC）：过滤低质量基因/细胞
    3. ★关键★：将原始 UMI 计数单独存入 layers["counts"]
                （scVI 强制要求原始 count，不能是 log 归一化的！）
    4. log 归一化 → 仅用于后续可视化
    5. 高可变基因（HVG）筛选 → 减少运算量，保留生物信号
    6. setup_anndata → 告诉 scVI 哪些列是批次/协变量/原始 count

【为什么原始 count 这么重要？】
    scVI 用的是 Zero-Inflated Negative Binomial (ZINB) 分布来建模 count 数据
    它会从原始整数计数中"反推"测序深度、技术噪声、生物学差异
    如果你喂 log(x+1)，它会把这当成连续值处理，模型彻底失效！

【为什么 HVG 选 1200？】
    - 太少：丢失 marker 基因
    - 太多：噪声大，训练慢
    - 1200 是单细胞分析常用折中
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import torch
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==================== 路径配置 ====================
# 当前在 code_interpretation/ 子目录，原始数据在上级目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # code_interpretation/
PARENT_DIR = os.path.dirname(BASE_DIR)                         # scvi-tools-learning/
DATA_PATH = os.path.join(PARENT_DIR, "heart_atlas.h5ad")
OUTPUT_DIR = os.path.join(PARENT_DIR, "code_interpretation_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 沿用模块1的设置
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=120, dpi_save=300, figsize=(6, 5),
                              facecolor='white', format='png')
sc.settings.figdir = './figures/'
os.makedirs(sc.settings.figdir, exist_ok=True)


# ============================================================
# 步骤 2.1：加载 h5ad 数据
# ============================================================
print("=" * 70)
print("模块2：数据载入与预处理")
print("=" * 70)

print("\n[2.1] 加载 h5ad 数据...")
adata = sc.read_h5ad(DATA_PATH)
# h5ad 是 AnnData 的标准存储格式（HDF5）
# 它包含三层数据：
#   adata.X     - 表达矩阵（默认 log-normalized）
#   adata.obs   - 细胞元信息（行：细胞）
#   adata.var   - 基因元信息（列：基因）
#   adata.obsm  - 细胞级多维注释（如 UMAP 坐标）
#   adata.layers - 额外数据层（如原始 counts）

print(f"  原始数据: {adata.shape[0]} 细胞 × {adata.shape[1]} 基因")
print(f"  obs 列: {list(adata.obs.columns)[:5]}...")  # 显示前5列
print(f"  cell_type 分布: \n{adata.obs['cell_type'].value_counts().head()}")


# ============================================================
# 步骤 2.2：质量控制 (QC) - 过滤低质量基因和细胞
# ============================================================
print("\n[2.2] 质量控制 (QC) ...")
# 为什么要过滤？
#   - 极低 counts 的基因往往是测序噪声（随机 spike-in）
#   - 只检测到 < 100 基因的细胞可能是死细胞或空液滴

sc.pp.filter_genes(adata, min_cells=3)
# 过滤掉在少于 3 个细胞中表达的基因（典型阈值 3-10）
# min_cells=3 是最保守的，能保留尽可能多信息

sc.pp.filter_cells(adata, min_genes=100)
# 过滤掉检测到 < 100 基因的细胞
# 心肌细胞基因表达丰度高，所以这个阈值不算严

print(f"  过滤后: {adata.shape[0]} 细胞 × {adata.shape[1]} 基因")


# ============================================================
# 步骤 2.3：★关键★ 保存原始 counts 到 layers["counts"]
# ============================================================
print("\n[2.3] 保存原始 UMI 计数到 layers['counts'] ...")
adata.layers["counts"] = adata.X.copy()
# adata.X 当前是 log-normalized 数据（被标准库自动处理过）
# 但 scVI 必须用原始整数 count，否则模型失效
# 所以我们把当前 X 复制到 layers["counts"]
# 之后 setup_anndata(layer="counts") 会让 scVI 从这里读取

print(f"  layers['counts'] 形状: {adata.layers['counts'].shape}")
print(f"  数据类型: {adata.layers['counts'].dtype}")
print(f"  示例值（前5个非零）: {adata.layers['counts'].data[:5]}")


# ============================================================
# 步骤 2.4：log 归一化（仅用于绘图，不用于 scVI 训练）
# ============================================================
print("\n[2.4] log 归一化（用于绘图） ...")
sc.pp.normalize_total(adata, target_sum=1e4)
# normalize_total 把每个细胞的文库大小归一化到 1e4
# 这样做的目的是消除测序深度差异（细胞 A 总 counts=5000，细胞 B 总 counts=20000）
# 归一化后两者可比

sc.pp.log1p(adata)
# log(x+1) 变换，让数据接近正态分布
# 适用于后续 UMAP/聚类等基于高斯假设的算法

adata.raw = adata  # 保存到 .raw，后续画 marker 基因用
# .raw 是个"备份"，画图时可以用 adata.raw[:, gene] 拿到 log-normalized 值

print(f"  log 归一化后: 范围 [{adata.X.min():.2f}, {adata.X.max():.2f}]")


# ============================================================
# 步骤 2.5：★重要★ 筛选高可变基因 HVG
# ============================================================
print("\n[2.5] 高可变基因筛选 (HVG) ...")
# 为什么只取 HVG？
#   - 基因数 1200 中很多是 housekeeping（不变化的）
#   - 全部用：维度灾难，训练慢 5-10 倍
#   - 仅 HVG：保留生物信号，剔除噪声

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,        # 保留变异最大的 1200 个基因
    flavor='seurat',         # Seurat 方法（基于 mean-variance 关系）
    batch_key='donor',       # ★关键★ 按 donor 分批计算 HVG
    subset=True              # 立即子集化：adata 只剩 1200 基因
)
# batch_key='donor' 让 HVG 选择在每个 donor 内变异大的基因
# 这能避免"批次基因"被误选为 HVG

print(f"  HVG 筛选后: {adata.shape[0]} 细胞 × {adata.shape[1]} 基因")


# ============================================================
# 步骤 2.6：★最关键★ setup_anndata
# ============================================================
print("\n[2.6] setup_anndata - 告诉 scVI 数据结构 ...")
# setup_anndata 是 scVI 的"注册"步骤
# 它扫描 adata 的元信息，建立数据管线
# 训练时 scVI 通过这个注册信息找到需要的数据

scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="donor",                       # ★关键★ 批次（不同供体带来的技术差异）
    # scVI 会学习去除 donor 间的批次效应
    # 一个 donor = 一个 batch
    categorical_covariate_keys=["cell_source"],  # 分类协变量（不需要去除，但建模考虑）
    # cell_source 是测序技术差异（如 Harvard-Nuclei vs Sanger-Cells）
    layer="counts",                          # ★关键★ 原始 count 在这里
    # 之前存入的 layers["counts"] 路径
    continuous_covariate_keys=["percent_mito", "percent_ribo"],  # 连续协变量
    # 让 scVI 知道哪些基因是"线粒体/核糖体"基因
    # 它会考虑这些干扰因素，但不会强行去除（因为有些细胞类型本来 mito 就高）
)
print("  setup_anndata 完成")
print(f"  批次: donor ({adata.obs['donor'].nunique()} 个)")
print(f"  分类协变量: cell_source ({adata.obs['cell_source'].nunique()} 个)")
print(f"  连续协变量: percent_mito, percent_ribo")

# ★关键★ 让 setup_anndata 信息持久化到 .uns
# 这样保存到 h5ad 后，重新加载时 setup 信息还在
adata.uns['_scvi_uuid'] = adata.uns['_scvi_manager_uuid']  # 触发 manager 保存
# 实际上 scanpy/scvi 在 setup 时会自动写 _scvi_manager_uuid 到 uns
# 但 h5ad 保存时会包含这个字段
# 我们先检查是否成功
assert '_scvi_manager_uuid' in adata.uns, "setup_anndata 信息未保存"
print(f"  scVI manager UUID: {adata.uns['_scvi_manager_uuid']}")


# ============================================================
# 步骤 2.7：绘制预处理 QC 图
# ============================================================
print("\n[2.7] 绘制预处理 QC 图 ...")
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
# 计算 n_genes_by_counts, total_counts, pct_counts_mt 等指标

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(adata.obs['total_counts'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Total counts'); axes[0].set_ylabel('Number of cells')
axes[0].set_title('Total counts per cell')

axes[1].hist(adata.obs['n_genes_by_counts'], bins=50, color='coral', edgecolor='white')
axes[1].set_xlabel('Genes per cell'); axes[1].set_ylabel('Number of cells')
axes[1].set_title('Genes per cell')

axes[2].hist(adata.obs['percent_mito'], bins=50, color='seagreen', edgecolor='white')
axes[2].set_xlabel('Percent mito'); axes[2].set_ylabel('Number of cells')
axes[2].set_title('Mitochondrial gene %')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_preprocess_qc.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  保存: {OUTPUT_DIR}/01_preprocess_qc.png")

# 保存中间结果（给后续模块用）
adata.write_h5ad(os.path.join(OUTPUT_DIR, "processed_adata.h5ad"))
print(f"  中间结果已保存: processed_adata.h5ad")
