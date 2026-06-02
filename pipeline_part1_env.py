"""
Heart Atlas Complete scVI Analysis Pipeline
6大模块完整分析：环境→预处理→scVI训练→提取结果→降维聚类→差异表达Marker
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
from matplotlib import rcParams

# ============================================================
# 模块1: 环境初始化
# ============================================================
print("=" * 70)
print("模块1: 环境初始化")
print("=" * 70)

# 固定随机种子
np.random.seed(0)
torch.manual_seed(0)

# 关闭显存回收警告
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# 配置Scanpy
sc.settings.verbosity = 1
sc.settings.set_figure_params(
    dpi=120, dpi_save=300,
    figsize=(6, 5),
    facecolor='white',
    vector_friendly=True,
    format='png'
)
sc.settings.figdir = './figures/'
os.makedirs(sc.settings.figdir, exist_ok=True)

# 检查 GPU
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
print(f"  Random seed: 0")
print(f"  Device: {device}")
print(f"  PyTorch: {torch.__version__}")
print(f"  scvi-tools: {scvi.__version__}")
print(f"  Output directory: {sc.settings.figdir}")
