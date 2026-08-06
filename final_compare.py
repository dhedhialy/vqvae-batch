import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_lisi(z, labels, n_neighbors=30):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(z)
    _, indices = nn.kneighbors(z)
    scores = []
    for i in range(len(z)):
        neighbor_labels = labels[indices[i]]
        _, counts = np.unique(neighbor_labels, return_counts=True)
        p = counts / n_neighbors
        scores.append(1.0 / np.sum(p ** 2))
    return float(np.mean(scores))


def cross_batch_ct_acc(z, batches, cell_types):
    from sklearn.ensemble import RandomForestClassifier
    unique_batches = np.unique(batches)
    scores = []
    for b in unique_batches:
        test_mask = batches == b
        train_mask = ~test_mask
        if test_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(z[train_mask], cell_types[train_mask])
        scores.append(rf.score(z[test_mask], cell_types[test_mask]))
    return float(np.mean(scores)), float(np.std(scores))


OUTPUT = "/home/aly/vqvae-batch/output/final"
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 65)
print("FINAL COMPARISON: scVI vs scVI+Harmony vs Scanorama vs scANVI")
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

import anndata as ad
import scanpy as sc
import harmonypy
import scvi

results = {}
random_bl = float(np.log(n_batches) / np.log(2))

print("\n[2] scVI (default: latent=30, layers=1)")
adata = ad.AnnData(X=X.copy())
adata.obs["batch"] = batches.astype(str)
adata.obs["cell_type"] = cell_types.astype(str)
adata.layers["counts"] = X.copy()

scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
model_scvi = scvi.model.SCVI(adata)
t0 = time.time()
model_scvi.train(max_epochs=300, early_stopping=True, early_stopping_patience=20)
print(f"  scVI trained in {time.time()-t0:.0f}s")

z_scvi = model_scvi.get_latent_representation()
bl1 = compute_lisi(z_scvi, batches)
cl1 = compute_lisi(z_scvi, cell_types)
ca1, cs1 = cross_batch_ct_acc(z_scvi, batches, cell_types)
print(f"  scVI alone:  batch_LISI={bl1:.3f}  celltype_LISI={cl1:.3f}  cross_CT={ca1:.3f}+/-{cs1:.3f}")
results["scvi_raw"] = {"batch_lisi": bl1, "celltype_lisi": cl1, "cross_batch_acc": ca1}

# scVI + Harmony
ho = harmonypy.run_harmony(z_scvi, {"batch": batches}, "batch", max_iter_harmony=50, theta=2.0)
z_sh = ho.Z_corr
bl_h = compute_lisi(z_sh, batches)
cl_h = compute_lisi(z_sh, cell_types)
ca_h, cs_h = cross_batch_ct_acc(z_sh, batches, cell_types)
print(f"  scVI+Harmony: batch_LISI={bl_h:.3f}  celltype_LISI={cl_h:.3f}  cross_CT={ca_h:.3f}+/-{cs_h:.3f}")
results["scvi_harmony"] = {"batch_lisi": bl_h, "celltype_lisi": cl_h, "cross_batch_acc": ca_h}

del model_scvi

print("\n[3] scANVI (scVI with cell type supervision)")
model_scvi2 = scvi.model.SCVI(adata, n_latent=30, n_layers=1)
model_scvi2.train(max_epochs=200, early_stopping=True, early_stopping_patience=20)
model_scanvi = scvi.model.SCANVI.from_scvi_model(model_scvi2, labels_key="cell_type", unlabeled_category="Unknown")
t0 = time.time()
model_scanvi.train(max_epochs=100, early_stopping=True, early_stopping_patience=10)
print(f"  scANVI trained in {time.time()-t0:.0f}s")

z_scanvi = model_scanvi.get_latent_representation()
bl2 = compute_lisi(z_scanvi, batches)
cl2 = compute_lisi(z_scanvi, cell_types)
ca2, cs2 = cross_batch_ct_acc(z_scanvi, batches, cell_types)
print(f"  scANVI alone: batch_LISI={bl2:.3f}  celltype_LISI={cl2:.3f}  cross_CT={ca2:.3f}+/-{cs2:.3f}")
results["scanvi_raw"] = {"batch_lisi": bl2, "celltype_lisi": cl2, "cross_batch_acc": ca2}

# scANVI + Harmony
ho2 = harmonypy.run_harmony(z_scanvi, {"batch": batches}, "batch", max_iter_harmony=50, theta=2.0)
z_sh2 = ho2.Z_corr
bl_h2 = compute_lisi(z_sh2, batches)
cl_h2 = compute_lisi(z_sh2, cell_types)
ca_h2, cs_h2 = cross_batch_ct_acc(z_sh2, batches, cell_types)
print(f"  scANVI+Harmony: batch_LISI={bl_h2:.3f}  celltype_LISI={cl_h2:.3f}  cross_CT={ca_h2:.3f}+/-{cs_h2:.3f}")
results["scanvi_harmony"] = {"batch_lisi": bl_h2, "celltype_lisi": cl_h2, "cross_batch_acc": ca_h2}

del model_scvi2, model_scanvi

print("\n[4] Scanorama (anchor-based integration)")
import scanorama
batch_names = np.unique(batches)
batch_indices = {b: np.where(batches == b)[0] for b in batch_names}
datasets = [X[batch_indices[b]] for b in batch_names]
gene_list = [list(range(X.shape[1])) for _ in batch_names]
t0 = time.time()
integrated, _ = scanorama.correct(datasets, gene_list, return_dimred=True)
print(f"  Scanorama integrated in {time.time()-t0:.0f}s")

# Reconstruct full matrix
z_scanorama = np.zeros((X.shape[0], integrated[0].shape[1]))
for i, b in enumerate(batch_names):
    z_scanorama[batch_indices[b]] = integrated[i]

bl3 = compute_lisi(z_scanorama, batches)
cl3 = compute_lisi(z_scanorama, cell_types)
ca3, cs3 = cross_batch_ct_acc(z_scanorama, batches, cell_types)
print(f"  Scanorama:     batch_LISI={bl3:.3f}  celltype_LISI={cl3:.3f}  cross_CT={ca3:.3f}+/-{cs3:.3f}")
results["scanorama"] = {"batch_lisi": bl3, "celltype_lisi": cl3, "cross_batch_acc": ca3}

print("\n" + "=" * 65)
print("FINAL COMPARISON TABLE")
print("=" * 65)
print(f"{'Method':<25s} {'batch_LISI':>10s} {'ct_LISI':>8s} {'cross_CT':>8s}")
print("-" * 55)
print(f"{'Random baseline':<25s} {random_bl:>10.3f}")
print(f"{'Perfect mixing':<25s} {float(n_batches):>10.3f}")
print("-" * 55)
for name, r in sorted(results.items(), key=lambda x: -x[1].get("batch_lisi", 0)):
    print(f"{name:<25s} {r['batch_lisi']:>10.3f} {r['celltype_lisi']:>8.3f} {r.get('cross_batch_acc', 0):>8.3f}")
print("-" * 55)

results["random_baseline"] = random_bl
results["perfect_mixing"] = float(n_batches)

with open(f"{OUTPUT}/final_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT}/final_results.json")

print("\nGenerating UMAPs...")
sc.settings.figdir = OUTPUT
for name, z_emb, title in [
    ("scvi", z_scvi, "scVI"),
    ("scvi_harmony", z_sh, "scVI + Harmony"),
    ("scanvi", z_scanvi, "scANVI"),
    ("scanorama", z_scanorama, "Scanorama"),
]:
    adata_umap = ad.AnnData(z_emb.astype(np.float32))
    adata_umap.obs["batch"] = batches.astype(str)
    adata_umap.obs["cell_type"] = cell_types.astype(str)
    sc.pp.neighbors(adata_umap, use_rep="X")
    sc.tl.umap(adata_umap)
    sc.pl.umap(adata_umap, color=["batch", "cell_type"], title=[f"{title} Batch", f"{title} Cell Type"],
               show=False)
    import matplotlib.pyplot as plt
    plt.savefig(f"{OUTPUT}/umap_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved umap_{name}.png")
