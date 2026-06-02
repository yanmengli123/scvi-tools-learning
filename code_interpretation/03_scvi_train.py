"""
================================================================================
模块3：SCVI 模型构建与训练 - 深度学习核心步骤
================================================================================
【scVI 模型本质】
    变分自编码器（VAE, Variational Autoencoder）:
    - 编码器: count → 30 维隐空间（μ, σ）
    - 解码器: 30 维隐空间 → 重建 count（ZINB 分布）
    - 目标: 让重建的 count 尽量接近真实 count
    - 同时: 隐空间不包含批次信息（通过对抗性 batch 建模去除）

【ZINB 分布 - 为什么单细胞数据用这个？】
    单细胞 RNA-seq 计数有三个特点：
    1. 都是非负整数
    2. 极度零膨胀（dropout）：很多真实表达基因测序到 0
    3. 高度过离散：方差 > 均值
    ZINB = 零膨胀 + 负二项，同时刻画这两个特点

【关键超参数】
    n_latent=30  : 隐空间维度（经验值 10-50）
    n_hidden=128 : 隐藏层宽度
    n_layers=2   : 编码器/解码器层数
    dispersion   : 色散参数学习方式
    gene_likelihood : 似然函数（nb=负二项, zinb=零膨胀负二项）
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import scanpy as sc
import scvi
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
# 步骤 3.1：加载预处理数据
# ============================================================
print("=" * 70)
print("模块3：SCVI 模型构建与训练")
print("=" * 70)

adata = sc.read_h5ad(os.path.join(OUTPUT_DIR, "processed_adata.h5ad"))
print(f"\n[3.0] 加载数据: {adata.shape}")


# ============================================================
# 步骤 3.2：构建 SCVI 模型
# ============================================================
print("\n[3.1] 构建 SCVI 模型 (变分自编码器)...")
# SCVI 内部结构：
#   encoder: count (n_cells × n_genes)
#          → hidden (n_cells × 128)  [2层 MLP]
#          → μ, log_var (n_cells × 30)
#   latent:  z = μ + σ * ε,  ε ~ N(0,1)   (重参数化技巧)
#   decoder: z (n_cells × 30)
#          → hidden (n_cells × 128)       [2层 MLP]
#          → ZINB 参数 (n_cells × n_genes × 3)  [mean, dispersion, dropout]

model = scvi.model.SCVI(
    adata,                       # 必须是经过 setup_anndata 注册的 AnnData
    n_latent=30,                 # 隐空间维度
    # 太小：信息压缩过度
    # 太大：保留噪声，批次效应难去除
    # 30 是单细胞分析的常用折中（10-50 都行）

    n_layers=2,                  # 编码器/解码器的层数
    # 1层：欠拟合
    # 2层：推荐
    # 3+层：容易过拟合，训练慢

    n_hidden=128,                # 隐藏层神经元数
    # 太小：模型容量不够
    # 太大：训练慢，可能过拟合
    # 128 是经验值（64/256 也常见）

    dropout_rate=0.1,            # Dropout 比例
    # 0.1 意味着每层有 10% 神经元随机失活
    # 防止过拟合，提升泛化能力

    dispersion="gene-batch",     # 色散参数
    # "gene": 每基因一个色散（默认）
    # "gene-batch": 每基因每批次一个色散（推荐，批次差异影响方差）
    # "gene-label": 每基因每细胞类型一个色散

    gene_likelihood="nb",        # 似然函数
    # "nb": 负二项分布（推荐，标准）
    # "zinb": 零膨胀负二布
    # "poisson": 泊松分布（最简单，但拟合差）
)
print(f"  模型参数:")
print(f"    n_latent=30, n_layers=2, n_hidden=128")
print(f"    dispersion='gene-batch', gene_likelihood='nb' (Negative Binomial)")


# ============================================================
# 步骤 3.3：训练模型
# ============================================================
print("\n[3.2] 训练模型 (自动 GPU 加速)...")

model.train(
    max_epochs=100,              # 最大训练轮数
    # 1 epoch = 整个数据集过一遍
    # 100 通常足够；复杂数据可加到 200-400

    batch_size=128,              # 每批 128 个细胞
    # 太小：梯度噪声大
    # 太大：GPU 显存不够

    train_size=0.9,              # 90% 训练，10% 验证
    # 验证集用来 early stopping

    early_stopping=True,         # 早停
    # 验证集 loss 不再下降时自动停止
    # 防止过拟合

    enable_progress_bar=False,   # 关闭进度条（控制台更干净）

    accelerator="auto",          # 自动选 GPU/CPU
    # GPU 速度是 CPU 的 5-10 倍
)
print("  训练完成!")


# ============================================================
# 步骤 3.4：绘制训练曲线
# ============================================================
print("\n[3.3] 绘制训练曲线 ...")
# 训练曲线是判断模型质量的关键
# 好的训练：loss 平稳下降，验证集 loss 接近训练集

train_history = model.history
# history 是个 dict，键如 'elbo_train', 'reconstruction_loss_train', 'kl_train' 等

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 左图：训练损失
if 'elbo_train' in train_history:
    axes[0].plot(train_history['elbo_train'], label='ELBO (train)', color='steelblue', linewidth=2)
if 'reconstruction_loss_train' in train_history:
    axes[0].plot(train_history['reconstruction_loss_train'], label='Reconstruction Loss', color='coral', linewidth=2)
if 'kl_train' in train_history:
    axes[0].plot(train_history['kl_train'], label='KL Divergence', color='seagreen', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Components', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# 右图：验证损失
if 'elbo_validation' in train_history:
    axes[1].plot(train_history['elbo_validation'], label='ELBO (val)', color='coral', linewidth=2)
if 'reconstruction_loss_validation' in train_history:
    axes[1].plot(train_history['reconstruction_loss_validation'], label='Recon Loss (val)', color='steelblue', linewidth=2)
if 'kl_validation' in train_history:
    axes[1].plot(train_history['kl_validation'], label='KL (val)', color='seagreen', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Validation Loss Components', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_training_curve.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  保存: {OUTPUT_DIR}/02_training_curve.png")


# ============================================================
# 步骤 3.5：保存模型（重要：可重复使用）
# ============================================================
print("\n[3.4] 保存模型 ...")
model_dir = os.path.join(OUTPUT_DIR, "scvi_heart_model")
model.save(model_dir + "/", overwrite=True)
# 保存的是：
#   - model_params.pt: 神经网络权重
#   - attr.pkl: 模型超参数
#   - var_names.csv: 基因顺序
# 下次可以直接 load，无需重训

print(f"  模型已保存: {model_dir}/")
print(f"  文件列表:")
for f in os.listdir(model_dir):
    size = os.path.getsize(os.path.join(model_dir, f)) / 1024
    print(f"    {f} ({size:.1f} KB)")
