"""
================================================================================
模块1：环境初始化 - scVI 分析的第一步
================================================================================
【核心目的】
    1. 固定随机种子 → 保证实验可复现（任何人都能跑出相同结果）
    2. 配置日志/警告 → 避免控制台刷屏
    3. 配置 Scanpy 出图参数 → 所有图自动高分辨率、自动保存
    4. 检测 GPU → 深度学习必须 GPU 加速

【本模块不涉及任何数据处理，仅是"准备工作"】
================================================================================
"""

import os                          # 操作系统接口（创建文件夹）
import numpy as np                 # 数值计算（随机种子）
import pandas as pd                # 表格数据处理（差异表达结果）
import torch                       # PyTorch 深度学习框架（scVI 底层）
import scanpy as sc                # 单细胞分析标准库（读 h5ad、画图、聚类）
import scvi                        # scvi-tools 库（单细胞深度学习工具包）
import matplotlib.pyplot as plt    # 基础绘图库

# ---------------------- 1. 抑制无关警告 ----------------------
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告，让控制台只显示关键信息

import logging                     # 日志系统
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# scVI 底层用 PyTorch Lightning 训练；它会打印很多 epoch 进度条
# 我们手动管理训练过程，所以把这些输出级别降为 ERROR（只显示错误）

# ---------------------- 2. 固定随机种子（可复现性核心） ----------------------
SEED = 0                           # 选 0 只是约定，可选 42 等任意整数
np.random.seed(SEED)               # numpy 随机数固定（如 KMeans 初始化、HVG 抽样）
torch.manual_seed(SEED)            # PyTorch 随机数固定（模型权重初始化、Dropout）

# ---------------------- 3. 配置 Scanpy 全局出图参数 ----------------------
sc.settings.verbosity = 1          # 0=静默, 1=进度, 2=详细；1 是黄金平衡点

sc.settings.set_figure_params(
    dpi=120,                       # 屏幕查看分辨率
    dpi_save=300,                  # 保存图片分辨率（论文级）
    figsize=(6, 5),                # 默认图大小（英寸）
    facecolor='white',             # 白色背景（更专业）
    vector_friendly=True,          # 避免透明背景（部分 PDF 渲染失败）
    format='png'                   # 默认保存 png
)

# ---------------------- 4. 配置输出目录 ----------------------
sc.settings.figdir = './figures/'   # 图片统一保存到这里
os.makedirs(sc.settings.figdir, exist_ok=True)  # exist_ok=True 表示目录已存在不报错

# ---------------------- 5. 检测硬件 ----------------------
use_gpu = torch.cuda.is_available()           # 是否有 NVIDIA GPU
device = "cuda" if use_gpu else "cpu"          # 选择设备

print("=" * 70)
print("模块1：环境初始化")
print("=" * 70)
print(f"  随机种子: {SEED}                    # 改种子能改变结果，但每次跑同一种子结果一致")
print(f"  设备: {device}                          # cuda=GPU加速, cpu=纯CPU（慢10倍）")
print(f"  PyTorch: {torch.__version__}")
print(f"  scvi-tools: {scvi.__version__}")
print(f"  图片输出目录: {sc.settings.figdir}")
print(f"  图片分辨率: {sc.settings.set_figure_params.__doc__ and '300 DPI'}")
