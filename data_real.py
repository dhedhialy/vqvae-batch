"""Real-cellxgene data loader fork for NB-based VQ-VAE.

Same batch/type filtering parameters as the server's load_cellxgene_data, but the
model receives RAW integer counts (from .raw) — which is what a Negative Binomial
reconstruction likelihood requires — instead of the log-normalized X that the
scVI baselines read. HVG selection is computed on log1p-normalized counts (the
standard scRNA-seq preprocessing) so the gene set used is meaningful.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class SingleCellDataset(Dataset):
    def __init__(self, X, batch_labels, cell_types=None):
        self.X = torch.FloatTensor(X)
        self.batch_labels = torch.LongTensor(batch_labels)
        self.cell_types = (
            torch.LongTensor(cell_types) if cell_types is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx], "batch": self.batch_labels[idx]}
        if self.cell_types is not None:
            item["cell_type"] = self.cell_types[idx]
        return item


def _as_dense(M):
    from scipy.sparse import issparse
    return np.asarray(M.toarray() if issparse(M) else M, dtype=np.float32)


def load_cellxgene_counts(
    h5ad_path,
    batch_key="batch",
    celltype_key="cell_type",
    min_cells_per_batch=20,
    min_cells_per_type=10,
    n_top_genes=2000,
    max_batches=20,
    max_cell_types=20,
):
    import anndata as ad
    from sklearn.preprocessing import LabelEncoder

    print(f"Loading {h5ad_path}...")
    a = ad.read_h5ad(h5ad_path)
    print(f"  Raw shape: {a.shape}")

    for k in (batch_key, celltype_key):
        if k not in a.obs.columns:
            raise ValueError(f"key '{k}' not found: {list(a.obs.columns)}")
    a.obs[batch_key] = a.obs[batch_key].astype(str)
    a.obs[celltype_key] = a.obs[celltype_key].astype(str)

    bc = a.obs[batch_key].value_counts()
    keep = bc[bc >= min_cells_per_batch].index.tolist()
    if len(keep) > max_batches:
        keep = bc.loc[keep].nlargest(max_batches).index.tolist()
    a = a[a.obs[batch_key].isin(keep)].copy()
    print(f"  cells after batch filter: {a.shape[0]} ({a.obs[batch_key].nunique()} batches)")

    tc = a.obs[celltype_key].value_counts()
    keep_t = tc[tc >= min_cells_per_type].index.tolist()
    if len(keep_t) > max_cell_types:
        keep_t = tc.loc[keep_t].nlargest(max_cell_types).index.tolist()
    a = a[a.obs[celltype_key].isin(keep_t)].copy()
    print(f"  cells after type filter: {a.shape[0]} ({a.obs[celltype_key].nunique()} types)")

    # ----- raw counts -----
    if a.raw is not None:
        counts = _as_dense(a.raw.X)
        print(f"  using .raw counts: {counts.shape}")
    elif "counts" in a.layers:
        counts = _as_dense(a.layers["counts"])
        print(f"  using layers['counts'] ({counts.shape})")
    else:
        counts = _as_dense(a.X)
        print("  using adata.X as counts")

    # ----- HVG on log1p-normalized counts (standard) -----
    if counts.shape[1] > n_top_genes:
        with np.errstate(divide="ignore", invalid="ignore"):
            lib = counts.sum(axis=1, keepdims=True)
            log_norm = np.log1p(counts / np.maximum(lib, 1) * 1e4)
        n_expr = (log_norm > 0).sum(axis=0)
        keep_g = n_expr >= 10  # expressed in >= 10 cells
        col = np.where(keep_g)[0]
        if col.size == 0:
            col = np.arange(counts.shape[1])
        vmean = log_norm[:, col].mean(axis=0)
        vvar = log_norm[:, col].var(axis=0)
        disp = (vvar - vmean) / np.maximum(vmean, 1e-9)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        hv = np.argsort(disp)[::-1][:n_top_genes]
        col = col[hv]
    else:
        col = np.arange(counts.shape[1])
    counts = counts[:, col]
    print(f"  genes after HVG: {counts.shape[1]}")

    n_before = counts.shape[0]
    valid = np.isfinite(counts).all(axis=1)
    counts = counts[valid]
    bl = a.obs[batch_key].values[valid]
    ct = a.obs[celltype_key].values[valid]
    if n_before - counts.shape[0]:
        print(f"  removed {n_before - counts.shape[0]} cells w/ NaN/Inf")

    b_enc = LabelEncoder(); bi = b_enc.fit_transform(bl)
    c_enc = LabelEncoder(); ci = c_enc.fit_transform(ct)
    print(f"  Final: {counts.shape[0]} x {counts.shape[1]}, "
          f"{len(b_enc.classes_)} batches, {len(c_enc.classes_)} cell types")
    return counts, bi, ci, {
        "n_batches": len(b_enc.classes_),
        "n_cell_types": len(c_enc.classes_),
        "n_genes": counts.shape[1],
        "batch_classes": b_enc.classes_.tolist(),
        "celltype_classes": c_enc.classes_.tolist(),
    }


def subsample_dataset(dataset, fraction=0.3, seed=42):
    rng = np.random.RandomState(seed)
    n = len(dataset)
    n_sub = max(100, int(n * fraction))
    idx = rng.choice(n, n_sub, replace=False)
    return Subset(dataset, idx)


def create_dataloaders(X, batch_labels, cell_types=None, batch_size=256, val_split=0.15, subsample=None):
    dataset = SingleCellDataset(X, batch_labels, cell_types)
    if subsample is not None:
        dataset = subsample_dataset(dataset, subsample)
    n = len(dataset)
    n_val = int(n * val_split)
    train, val = torch.utils.data.random_split(
        dataset, [n - n_val, n_val], generator=torch.Generator().manual_seed(42)
    )
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False), dataset)