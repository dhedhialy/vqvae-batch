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
