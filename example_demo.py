"""
scvi-tools Quick Demo - Simplified Output
"""

import warnings
warnings.filterwarnings("ignore")

import scvi
import numpy as np
from scvi.data import synthetic_iid

print("=" * 50)
print("scvi-tools Quick Demo")
print("=" * 50)

# 1. Generate synthetic data
print("\n[1] Generating synthetic data...")
adata = synthetic_iid(
    batch_size=100,
    n_genes=50,
    n_batches=2,
    n_labels=3,
)
print(f"  Data shape: {adata.shape}")
print(f"  Batches: {list(adata.obs['batch'].unique())}")
print(f"  Labels: {list(adata.obs['labels'].unique())}")

# 2. Setup AnnData
print("\n[2] Setting up AnnData...")
scvi.model.SCVI.setup_anndata(
    adata,
    batch_key="batch",
    labels_key="labels"
)

# 3. Train model
print("\n[3] Training SCVI model...")
model = scvi.model.SCVI(adata)
model.train(
    max_epochs=50,
    train_size=0.8,
    early_stopping=True,
    enable_progress_bar=False,  # Disable progress bar for cleaner output
)
print("  Training completed!")

# 4. Get latent representation
print("\n[4] Getting latent representation...")
latent = model.get_latent_representation()
print(f"  Latent shape: {latent.shape}")

# 5. Get denoised expression
print("\n[5] Getting denoised expression...")
denoised = model.get_normalized_expression()
print(f"  Denoised shape: {denoised.shape}")

# 6. Differential expression
print("\n[6] Running differential expression analysis...")
de_results = model.differential_expression(
    groupby="labels",
    group1="label_0",
    group2="label_1",
)
print(f"  Results shape: {de_results.shape}")
print("\n  Top 5 upregulated genes:")
print(de_results.head())

# 7. Save model
print("\n[7] Saving model...")
model.save("scvi_model/", overwrite=True)
print("  Model saved to scvi_model/")

print("\n" + "=" * 50)
print("Demo completed successfully!")
print("=" * 50)
