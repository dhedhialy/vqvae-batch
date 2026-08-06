import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = "/home/aly/vqvae-batch/output/fsq/cached_data.npz"

if os.path.exists(CACHE):
    print("Cache exists, skip")
    sys.exit(0)

print("Building data cache...", flush=True)
from data import load_cellxgene_data
X, batches, cell_types, meta = load_cellxgene_data(
    "/data/bhy/czcellxgene/h5ads/0b75c598-0893-4216-afe8-5414cab7739d.h5ad",
    batch_key="donor_id", celltype_key="cell_type",
    min_cells_per_batch=50, min_cells_per_type=50,
    n_top_genes=2000, max_batches=15, max_cell_types=20,
)
np.savez_compressed(
    CACHE,
    X=X, batches=batches, cell_types=cell_types,
    gene_names=np.array(meta["gene_names"]),
    n_batches=meta["n_batches"], n_cell_types=meta["n_cell_types"],
)
print(f"Saved cache: {CACHE}", flush=True)
