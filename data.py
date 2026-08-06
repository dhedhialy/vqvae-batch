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
        item = {
            "x": self.X[idx],
            "batch": self.batch_labels[idx],
        }
        if self.cell_types is not None:
            item["cell_type"] = self.cell_types[idx]
        return item


def load_cellxgene_data(
    h5ad_path,
    batch_key="batch",
    celltype_key="cell_type",
    min_cells_per_batch=20,
    min_cells_per_type=10,
    n_top_genes=2000,
    max_batches=20,
    max_cell_types=20,
    seed=42,
):
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required for real data loading: pip install anndata")
    import scanpy as sc
    from sklearn.preprocessing import LabelEncoder

    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Raw shape: {adata.shape}")

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found. Available: {list(adata.obs.columns)}")
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"Cell type key '{celltype_key}' not found. Available: {list(adata.obs.columns)}")

    adata.obs[batch_key] = adata.obs[batch_key].astype(str)
    adata.obs[celltype_key] = adata.obs[celltype_key].astype(str)

    batch_counts = adata.obs[batch_key].value_counts()
    keep_batches = batch_counts[batch_counts >= min_cells_per_batch].index.tolist()
    if len(keep_batches) > max_batches:
        keep_batches = batch_counts.loc[keep_batches].nlargest(max_batches).index.tolist()
    adata = adata[adata.obs[batch_key].isin(keep_batches)].copy()
    print(f"  After filtering batches (>= {min_cells_per_batch} cells, max {max_batches}): {adata.shape[0]} cells, {adata.obs[batch_key].nunique()} batches")

    ct_counts = adata.obs[celltype_key].value_counts()
    keep_types = ct_counts[ct_counts >= min_cells_per_type].index.tolist()
    if len(keep_types) > max_cell_types:
        keep_types = ct_counts.loc[keep_types].nlargest(max_cell_types).index.tolist()
    adata = adata[adata.obs[celltype_key].isin(keep_types)].copy()
    print(f"  After filtering cell types (>= {min_cells_per_type} cells, max {max_cell_types}): {adata.shape[0]} cells, {adata.obs[celltype_key].nunique()} cell types")

    if adata.X.shape[1] > n_top_genes:
        from scipy.sparse import issparse
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f"  After HVG selection: {adata.shape[1]} genes")
    else:
        from scipy.sparse import issparse
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

    n_before = adata.shape[0]
    X = np.array(adata.X, dtype=np.float32)
    valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[valid]
    batch_labels = adata.obs[batch_key].values[valid]
    cell_type_labels = adata.obs[celltype_key].values[valid]
    n_removed = n_before - X.shape[0]
    if n_removed > 0:
        print(f"  Removed {n_removed} cells with NaN/Inf values")

    batch_enc = LabelEncoder()
    batch_ints = batch_enc.fit_transform(batch_labels)
    n_batches = len(batch_enc.classes_)
    print(f"  Batch mapping: {dict(zip(batch_enc.classes_, range(n_batches)))}")

    ct_enc = LabelEncoder()
    ct_ints = ct_enc.fit_transform(cell_type_labels)
    n_cell_types = len(ct_enc.classes_)
    print(f"  Cell type mapping: {dict(zip(ct_enc.classes_, range(n_cell_types)))}")

    print(f"  Final: {X.shape[0]} cells x {X.shape[1]} genes, {n_batches} batches, {n_cell_types} cell types")
    # Gene names (use feature_name column if present, else var_names)
    if "feature_name" in adata.var.columns:
        gene_names = adata.var["feature_name"].astype(str).values
    else:
        gene_names = np.asarray(adata.var_names)
    return X, batch_ints, ct_ints, {
        "n_batches": n_batches,
        "n_cell_types": n_cell_types,
        "n_genes": X.shape[1],
        "batch_classes": batch_enc.classes_.tolist(),
        "celltype_classes": ct_enc.classes_.tolist(),
        "gene_names": gene_names.tolist(),
    }


def generate_synthetic_data(
    n_cells=5000,
    n_genes=2000,
    n_batches=5,
    n_cell_types=8,
    n_programs=20,
    seed=42,
    return_counts=True,
):
    rng = np.random.RandomState(seed)
    batch_assignments = rng.randint(0, n_batches, size=n_cells)
    cell_type_assignments = rng.randint(0, n_cell_types, size=n_cells)
    batch_offset = rng.randn(n_batches, n_genes) * 2.0
    genes_per_program = n_genes // n_programs
    program_strength = np.zeros((n_cell_types, n_programs))
    for ct in range(n_cell_types):
        active = rng.choice(n_programs, size=3, replace=False)
        program_strength[ct, active] = rng.uniform(2.0, 5.0, size=3)
    program_genes = np.zeros((n_programs, n_genes))
    for pg in range(n_programs):
        start = pg * genes_per_program
        end = start + genes_per_program if pg < n_programs - 1 else n_genes
        program_genes[pg, start:end] = rng.uniform(0.5, 1.5, size=end - start)
    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        base = program_strength[cell_type_assignments[i]] @ program_genes
        base += batch_offset[batch_assignments[i]]
        base += rng.randn(n_genes) * 0.3
        base = np.exp(np.clip(base, 0.01, 4.5))
        counts[i] = rng.poisson(np.clip(base, 0.01, 50)).astype(np.float32)
    lib_size = counts.sum(axis=1, keepdims=True)
    cpm = counts / np.maximum(lib_size, 1) * 10000
    log_cpm = np.log1p(cpm)
    high_var_idx = np.argsort(log_cpm.var(axis=0))[-1000:]
    if return_counts:
        return counts[:, high_var_idx].astype(np.float32), batch_assignments, cell_type_assignments
    return log_cpm[:, high_var_idx], batch_assignments, cell_type_assignments


def subsample_dataset(dataset, fraction=0.3, seed=42):
    rng = np.random.RandomState(seed)
    n = len(dataset)
    n_sub = max(100, int(n * fraction))
    indices = rng.choice(n, n_sub, replace=False)
    return Subset(dataset, indices)


def create_dataloaders(
    X, batch_labels, cell_types=None, batch_size=256, val_split=0.15, subsample=None
):
    dataset = SingleCellDataset(X, batch_labels, cell_types)
    if subsample is not None:
        dataset = subsample_dataset(dataset, subsample)
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val
    train, val = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset
