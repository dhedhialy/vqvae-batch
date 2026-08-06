import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT = "/home/aly/vqvae-batch/output/deep_dive"
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 65)
print("DEEP DIVE: scVI architecture & corrected expression analysis")
print("=" * 65)

print("\n[1] Loading data...")
from data import load_cellxgene_data
X, batches, cell_types, meta = load_cellxgene_data(
    "/data/bhy/czcellxgene/h5ads/0b75c598-0893-4216-afe8-5414cab7739d.h5ad",
    batch_key="donor_id", celltype_key="cell_type",
    min_cells_per_batch=50, min_cells_per_type=50,
    n_top_genes=2000, max_batches=15, max_cell_types=20,
)
n_genes, n_batches, n_ct = meta["n_genes"], meta["n_batches"], meta["n_cell_types"]
print(f"  {X.shape[0]} cells, {n_genes} genes, {n_batches} batches, {n_ct} cell types")

batch_names = sorted(np.unique(batches))
ct_names = sorted(np.unique(cell_types))

import anndata as ad
import scanpy as sc
import scvi
import harmonypy

# Setup data
adata = ad.AnnData(X=X.copy())
adata.obs["batch"] = batches.astype(str)
adata.obs["cell_type"] = cell_types.astype(str)
adata.layers["counts"] = X.copy()

print("\n[2] Training scVI...")
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
model_path = f"{OUTPUT}/scvi_model"
if os.path.exists(model_path):
    print(f"  Loading saved model from {model_path}")
    model = scvi.model.SCVI.load(model_path, adata)
else:
    model = scvi.model.SCVI(adata, n_latent=30, n_layers=2)
    t0 = time.time()
    model.train(max_epochs=300, early_stopping=True, early_stopping_patience=20)
    print(f"  Trained in {time.time()-t0:.0f}s")
    model.save(model_path, overwrite=True)
    print(f"  Saved model to {model_path}")

# --- Latent space evaluation ---
print("\n[3] Latent space evaluation...")
z = model.get_latent_representation()

def compute_lisi(z, labels):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=30, n_jobs=-1).fit(z)
    _, idx = nn.kneighbors(z)
    scores = []
    for i in range(len(z)):
        _, cnt = np.unique(labels[idx[i]], return_counts=True)
        p = cnt / 30
        scores.append(1.0 / np.sum(p ** 2))
    return float(np.mean(scores))

def cross_batch_ct(z, batch, ct):
    from sklearn.ensemble import RandomForestClassifier
    sc = []
    for b in np.unique(batch):
        te = batch == b; tr = ~te
        if te.sum() < 10: continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(z[tr], ct[tr]); sc.append(rf.score(z[te], ct[te]))
    return float(np.mean(sc))

bl_lat = compute_lisi(z, batches)
cl_lat = compute_lisi(z, cell_types)
ca_lat = cross_batch_ct(z, batches, cell_types)
print(f"  scVI latent: batch_LISI={bl_lat:.3f}  celltype_LISI={cl_lat:.3f}  cross_CT={ca_lat:.3f}")

# scVI + Harmony
ho = harmonypy.run_harmony(z, {"batch": batches}, "batch", max_iter_harmony=50)
z_harmony = ho.Z_corr
bl_har = compute_lisi(z_harmony, batches)
cl_har = compute_lisi(z_harmony, cell_types)
ca_har = cross_batch_ct(z_harmony, batches, cell_types)
print(f"  scVI+Harmony: batch_LISI={bl_har:.3f}  celltype_LISI={cl_har:.3f}  cross_CT={ca_har:.3f}")

# --- Corrected expression evaluation ---
print("\n[4] Batch-corrected expression (decode with reference batch)...")
# Valid batch categories straight from the model's state registry
batch_cats = list(model.adata_manager.get_state_registry("batch").categorical_mapping)
print(f"  Valid batch categories: {batch_cats}")
corrected_expr = model.get_normalized_expression(
    n_samples=1,
    transform_batch=batch_cats,  # average over all batches
    return_numpy=True,
)
print(f"  Corrected expression shape: {corrected_expr.shape}")

# Save for reference
np.save(f"{OUTPUT}/corrected_expr.npy", corrected_expr)
np.save(f"{OUTPUT}/raw_expr.npy", X)

# --- Per-gene analysis: batch vs cell type variance ---
print("\n[5] Per-gene variance decomposition...")

# For each gene, compute variance explained by batch and by cell type
def var_explained(data, labels):
    """Fraction of variance explained by group labels."""
    global_mean = np.mean(data)
    ss_total = np.sum((data - global_mean) ** 2)
    ss_between = 0
    for l in np.unique(labels):
        mask = labels == l
        group_mean = np.mean(data[mask])
        ss_between += len(data[mask]) * (group_mean - global_mean) ** 2
    return ss_between / max(ss_total, 1e-10)

# For raw expression
raw_batch_var = np.array([var_explained(X[:, i], batches) for i in range(n_genes)])
raw_ct_var = np.array([var_explained(X[:, i], cell_types) for i in range(n_genes)])

# For corrected expression
corr_batch_var = np.array([var_explained(corrected_expr[:, i], batches) for i in range(n_genes)])
corr_ct_var = np.array([var_explained(corrected_expr[:, i], cell_types) for i in range(n_genes)])

print(f"  Raw expression:")
print(f"    Mean batch variance explained: {np.mean(raw_batch_var):.4f}")
print(f"    Mean celltype variance explained: {np.mean(raw_ct_var):.4f}")
print(f"  Corrected expression:")
print(f"    Mean batch variance explained: {np.mean(corr_batch_var):.4f}")
print(f"    Mean celltype variance explained: {np.mean(corr_ct_var):.4f}")

# --- Visualize top batch-associated genes ---
print("\n[6] Visualizing top batch-associated genes...")
top_batch_genes = np.argsort(-raw_batch_var)[:6]  # top 6
top_ct_genes = np.argsort(-raw_ct_var)[:6]

fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i, gi in enumerate(top_batch_genes):
    ax = axes[0, i]
    for bidx, b in enumerate(batch_names):
        mask = batches == b
        ax.scatter(np.full(mask.sum(), bidx) + np.random.uniform(-0.2, 0.2, mask.sum()),
                   X[mask, gi], s=0.5, alpha=0.3, c=f"C{bidx % 10}")
    ax.set_title(f"Gene {gi}\n(batch var={raw_batch_var[gi]:.2f})")
    ax.set_xticks(range(len(batch_names)))
    ax.set_xlabel("Batch")

for i, gi in enumerate(top_ct_genes):
    ax = axes[1, i]
    for cidx, c in enumerate(ct_names):
        mask = cell_types == c
        ax.scatter(np.full(mask.sum(), cidx) + np.random.uniform(-0.2, 0.2, mask.sum()),
                   X[mask, gi], s=0.5, alpha=0.3, c=f"C{cidx % 16}")
    ax.set_title(f"Gene {gi}\n(ct var={raw_ct_var[gi]:.2f})")
    ax.set_xticks(range(len(ct_names)))
    ax.set_xlabel("Cell Type")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/raw_genes.png", dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  Saved raw_genes.png")

# Same for corrected expression
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i, gi in enumerate(top_batch_genes):
    ax = axes[0, i]
    for bidx, b in enumerate(batch_names):
        mask = batches == b
        ax.scatter(np.full(mask.sum(), bidx) + np.random.uniform(-0.2, 0.2, mask.sum()),
                   corrected_expr[mask, gi], s=0.5, alpha=0.3, c=f"C{bidx % 10}")
    ax.set_title(f"Gene {gi}\n(batch var={corr_batch_var[gi]:.2f})")
    ax.set_xticks(range(len(batch_names)))
    ax.set_xlabel("Batch")

for i, gi in enumerate(top_ct_genes):
    ax = axes[1, i]
    for cidx, c in enumerate(ct_names):
        mask = cell_types == c
        ax.scatter(np.full(mask.sum(), cidx) + np.random.uniform(-0.2, 0.2, mask.sum()),
                   corrected_expr[mask, gi], s=0.5, alpha=0.3, c=f"C{cidx % 16}")
    ax.set_title(f"Gene {gi}\n(ct var={corr_ct_var[gi]:.2f})")
    ax.set_xticks(range(len(ct_names)))
    ax.set_xlabel("Cell Type")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/corrected_genes.png", dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  Saved corrected_genes.png")

# --- Distribution of variance explained ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(raw_batch_var, bins=50, alpha=0.5, label="Raw", density=True)
axes[0].hist(corr_batch_var, bins=50, alpha=0.5, label="Corrected", density=True)
axes[0].set_xlabel("Variance explained by BATCH")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].axvline(np.mean(raw_batch_var), color="C0", ls="--", alpha=0.7)
axes[0].axvline(np.mean(corr_batch_var), color="C1", ls="--", alpha=0.7)

axes[1].hist(raw_ct_var, bins=50, alpha=0.5, label="Raw", density=True)
axes[1].hist(corr_ct_var, bins=50, alpha=0.5, label="Corrected", density=True)
axes[1].set_xlabel("Variance explained by CELL TYPE")
axes[1].legend()
axes[1].axvline(np.mean(raw_ct_var), color="C0", ls="--", alpha=0.7)
axes[1].axvline(np.mean(corr_ct_var), color="C1", ls="--", alpha=0.7)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/variance_decomposition.png", dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  Saved variance_decomposition.png")

# --- UMAP on corrected expression ---
print("\n[7] UMAP on corrected expression...")
adata_corr = ad.AnnData(corrected_expr.astype(np.float32))
adata_corr.obs["batch"] = batches.astype(str)
adata_corr.obs["cell_type"] = cell_types.astype(str)
sc.pp.neighbors(adata_corr)
sc.tl.umap(adata_corr)
sc.pl.umap(adata_corr, color=["batch", "cell_type"],
           title=["Corrected expression - Batch", "Corrected expression - Cell Type"],
           show=False)
plt.savefig(f"{OUTPUT}/umap_corrected_expr.png", dpi=150, bbox_inches="tight")
plt.close("all")
print(f"  Saved umap_corrected_expr.png")

# --- Summary ---
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"scVI latent space:")
print(f"  batch_LISI={bl_lat:.3f} | celltype_LISI={cl_lat:.3f} | cross_batch_CT={ca_lat:.3f}")
print(f"scVI + Harmony latent:")
print(f"  batch_LISI={bl_har:.3f} | celltype_LISI={cl_har:.3f} | cross_batch_CT={ca_har:.3f}")
print(f"\nExpression variance decomposition:")
print(f"  BATCH:    Raw={np.mean(raw_batch_var):.4f} → Corrected={np.mean(corr_batch_var):.4f} ({np.mean(raw_batch_var)/np.mean(corr_batch_var):.1f}x reduction)")
print(f"  CELLTYPE: Raw={np.mean(raw_ct_var):.4f} → Corrected={np.mean(corr_ct_var):.4f}")

results = {
    "latent": {"batch_lisi": bl_lat, "celltype_lisi": cl_lat, "cross_batch_ct": ca_lat},
    "latent_harmony": {"batch_lisi": bl_har, "celltype_lisi": cl_har, "cross_batch_ct": ca_har},
    "expression_raw": {"mean_batch_var": float(np.mean(raw_batch_var)), "mean_ct_var": float(np.mean(raw_ct_var))},
    "expression_corrected": {"mean_batch_var": float(np.mean(corr_batch_var)), "mean_ct_var": float(np.mean(corr_ct_var))},
}
with open(f"{OUTPUT}/deep_dive_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nAll results saved to {OUTPUT}/")
