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


OUTPUT = "/home/aly/vqvae-batch/output/scvi_push"
os.makedirs(OUTPUT, exist_ok=True)
all_results = {}

print("=" * 60)
print("PUSHING scVI: multiple configs, longer training")
print("=" * 60)

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

random_bl = float(np.log(n_batches) / np.log(2))

configs = [
    {"n_latent": 10, "n_layers": 1, "dropout_rate": 0.1, "name": "scvi_L10_l1"},
    {"n_latent": 30, "n_layers": 2, "dropout_rate": 0.1, "name": "scvi_L30_l2"},
    {"n_latent": 30, "n_layers": 3, "dropout_rate": 0.1, "name": "scvi_L30_l3"},
    {"n_latent": 50, "n_layers": 2, "dropout_rate": 0.1, "name": "scvi_L50_l2"},
]

for cfg in configs:
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"Config: {name} (latent={cfg['n_latent']}, layers={cfg['n_layers']})")
    print(f"{'='*60}")

    adata = ad.AnnData(X=X.copy())
    adata.obs["batch"] = batches.astype(str)
    adata.obs["cell_type"] = cell_types.astype(str)
    adata.layers["counts"] = X.copy()

    import scvi
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
    model = scvi.model.SCVI(
        adata,
        n_latent=cfg["n_latent"],
        n_layers=cfg["n_layers"],
        dropout_rate=cfg["dropout_rate"],
    )

    print(f"  Training scVI (400 epochs max, early stopping)...")
    t0 = time.time()
    model.train(max_epochs=400, early_stopping=True, early_stopping_patience=30, train_size=0.9)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s, epochs trained: {model.history['elbo_train'].shape[0] if hasattr(model.history.get('elbo_train', None), 'shape') else '?'}")

    z_scvi = model.get_latent_representation()
    print(f"  Latent shape: {z_scvi.shape}")

    batch_lisi = compute_lisi(z_scvi, batches)
    ct_lisi = compute_lisi(z_scvi, cell_types)
    ct_acc, ct_std = cross_batch_ct_acc(z_scvi, batches, cell_types)
    print(f"  scVI alone:  batch_LISI={batch_lisi:.3f}  celltype_LISI={ct_lisi:.3f}  cross_batch_CT={ct_acc:.3f}+/-{ct_std:.3f}")

    all_results[f"{name}_raw"] = {
        "batch_lisi": batch_lisi, "celltype_lisi": ct_lisi,
        "cross_batch_acc": ct_acc, "cross_batch_std": ct_std,
        "train_time": train_time,
    }

    for theta in [1.0, 4.0, 8.0]:
        ho = harmonypy.run_harmony(z_scvi, {"batch": batches}, "batch", max_iter_harmony=50, theta=theta)
        z_h = ho.Z_corr
        bl_h = compute_lisi(z_h, batches)
        cl_h = compute_lisi(z_h, cell_types)
        ca_h, cs_h = cross_batch_ct_acc(z_h, batches, cell_types)
        print(f"  +Harmony(theta={theta}): batch_LISI={bl_h:.3f}  celltype_LISI={cl_h:.3f}  cross_batch_CT={ca_h:.3f}+/-{cs_h:.3f}")
        all_results[f"{name}_harmony_t{theta}"] = {
            "batch_lisi": bl_h, "celltype_lisi": cl_h,
            "cross_batch_acc": ca_h, "cross_batch_std": cs_h,
        }

    del model
    import gc; gc.collect()

print("\n" + "=" * 60)
print("FINAL RESULTS (sorted by batch_LISI)")
print("=" * 60)
print(f"{'Method':<35s} {'batch_LISI':>10s} {'ct_LISI':>8s} {'cross_CT':>8s}")
print("-" * 65)
print(f"{'Random baseline':<35s} {random_bl:>10.3f}")
print(f"{'Perfect mixing':<35s} {float(n_batches):>10.3f}")
print("-" * 65)
sorted_results = sorted(all_results.items(), key=lambda x: -x[1].get("batch_lisi", 0))
for name, r in sorted_results:
    print(f"{name:<35s} {r['batch_lisi']:>10.3f} {r['celltype_lisi']:>8.3f} {r.get('cross_batch_acc', 0):>8.3f}")

best_name, best_r = sorted_results[0]
print(f"\nBEST: {best_name}")
print(f"  batch_LISI={best_r['batch_lisi']:.3f} (random={random_bl:.3f}, perfect={n_batches})")
print(f"  celltype_LISI={best_r['celltype_lisi']:.3f}")
print(f"  cross-batch CT accuracy={best_r.get('cross_batch_acc',0):.3f}")

with open(f"{OUTPUT}/results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {OUTPUT}/results.json")
