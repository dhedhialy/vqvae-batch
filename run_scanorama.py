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

print("[1] Loading data...")
from data import load_cellxgene_data
X, batches, cell_types, meta = load_cellxgene_data(
    "/data/bhy/czcellxgene/h5ads/0b75c598-0893-4216-afe8-5414cab7739d.h5ad",
    batch_key="donor_id", celltype_key="cell_type",
    min_cells_per_batch=50, min_cells_per_type=50,
    n_top_genes=2000, max_batches=15, max_cell_types=20,
)
print(f"  {X.shape[0]} cells, {meta['n_genes']} genes, {meta['n_batches']} batches")

import scanorama
from sklearn.decomposition import PCA

print(f"[2] Running Scanorama...")
batch_names = np.unique(batches)
datasets = [X[batches == b] for b in batch_names]
genes = [list(range(X.shape[1])) for _ in batch_names]

t0 = time.time()
res = scanorama.correct(datasets, genes, return_dimred=True)
print(f"  Done in {time.time() - t0:.0f}s")

# Result[1] = list of per-batch PCA embeddings (sparse)
# Stack them into full matrix
import scipy.sparse
z_list = []
for emb in res[1]:
    if scipy.sparse.issparse(emb):
        emb = emb.toarray()
    z_list.append(emb)
z_scanorama = np.vstack(z_list)
print(f"  Shape: {z_scanorama.shape}")
elif isinstance(res, list):
    # Check if elements are arrays or lists
    if hasattr(res[0], 'shape'):
        z_scanorama = np.vstack(res)
    elif isinstance(res[0], list) and hasattr(res[0][0], 'shape'):
        print(f"  List of lists, inner shapes: {[r[0].shape if hasattr(r[0], 'shape') else '?' for r in res]}")
        # Might be list of (corrected, pca) tuples
        z_scanorama = np.vstack([r[1] if len(r) > 1 else r[0] for r in res])

print(f"  Final shape: {z_scanorama.shape}")

print("[3] Evaluating...")
bl = compute_lisi(z_scanorama, batches)
cl = compute_lisi(z_scanorama, cell_types)
ca, cs = cross_batch_ct_acc(z_scanorama, batches, cell_types)

print(f"\n=== Scanorama Results ===")
print(f"  batch_LISI={bl:.3f}")
print(f"  celltype_LISI={cl:.3f}")
print(f"  cross_batch_CT={ca:.3f}+/-{cs:.3f}")

results = {"batch_lisi": bl, "celltype_lisi": cl, "cross_batch_acc": ca, "cross_batch_std": cs}
with open(f"{OUTPUT}/scanorama_result.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT}/scanorama_result.json")
