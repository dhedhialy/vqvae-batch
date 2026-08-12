"""Extract the representations the scorecard scores.

Every scorecard question is asked of the same cells under several views:

``input_expression``      normalized-log1p input (leakage upper bound)
``encoder_z_e``           continuous pre-quantization latent
``bio_z_q``               quantized latent, the biological representation
``bio_code_onehot``       the discrete 32x16 code choice itself
``technical_embedding``   the scVI-style technical branch embedding alone
``bio_reconstruction``    decoded biology without the technical offset
``full_reconstruction``   biology + technical offset (sanity check)

High-dimensional views are passed through one fixed, label-independent Gaussian
projection so neighbour searches stay affordable at atlas scale; the projection
is a linear map, so linear readability is preserved up to the usual
Johnson-Lindenstrauss distortion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from atlas_eval.paths import ensure_on_path

LABEL_FIELDS = (
    "cell_type",
    "coarse_cell_type",
    "tissue",
    "disease",
    "assay",
    "dataset_id",
    "donor_id",
    "sample_id",
    "batch_id",
    "technical_block",
    "suspension_type",
    "sex",
    "development_stage",
    "age",
)

DEFAULT_SPLITS = ("test",)


@dataclass
class RepresentationPack:
    representations: Dict[str, np.ndarray]
    labels: Dict[str, np.ndarray]
    code_indices: np.ndarray
    soft_code_usage: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.code_indices.shape[0])


class _Projector:
    """One fixed Gaussian projection per (name, input dim), created lazily on device."""

    def __init__(self, dim: int, seed: int, device: torch.device) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.device = device
        self._matrices: Dict[str, torch.Tensor] = {}

    def __call__(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.dim <= 0 or tensor.shape[1] <= self.dim:
            return tensor
        key = f"{name}:{tensor.shape[1]}"
        if key not in self._matrices:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + (hash(key) % 100_000))
            matrix = torch.randn(tensor.shape[1], self.dim, generator=generator) / np.sqrt(self.dim)
            self._matrices[key] = matrix.to(self.device, dtype=tensor.dtype)
        return tensor @ self._matrices[key]


def load_atlas_model(
    config_path: str,
    run_id: str,
    checkpoint: str = "best",
    vq2608_root: Optional[str] = None,
):
    """Load the trained atlas model exactly as the project's own evaluators do."""
    ensure_on_path(vq2608_root)
    from vq_pipeline.checkpoints import checkpoint_metadata, load_checkpoint_and_config
    from vq_pipeline.runtime import load_config, make_output_dirs, pick_device

    base_config = load_config(config_path)
    out = make_output_dirs(base_config, run_id)
    ckpt, config, checkpoint_path = load_checkpoint_and_config(
        config_path, out["root"], checkpoint_name=checkpoint, return_path=True
    )
    device = pick_device(config["training"]["device"])
    meta = checkpoint_metadata(ckpt, config, checkpoint_path)
    return ckpt, config, device, out, meta


def load_bundle(config: Dict[str, Any], data_run_id: str, vq2608_root: Optional[str] = None) -> Dict[str, Any]:
    ensure_on_path(vq2608_root)
    from vq_pipeline.bundle import load_data_bundle

    local = dict(config)
    local["data"] = dict(config.get("data", {}))
    local["data"]["_data_run_id"] = data_run_id
    root = Path(config["output"]["root"]) / data_run_id
    path = root / "data_bundle.slim.json"
    if not path.exists():
        path = root / "data_bundle.json"
    if not path.exists():
        raise FileNotFoundError(f"No data bundle for {data_run_id} under {root}")
    return load_data_bundle(path, local)


def build_model_for_bundle(ckpt: Dict[str, Any], config: Dict[str, Any], bundle: Dict[str, Any], device: torch.device):
    ensure_on_path()
    from clean_common import build_model

    model = build_model(input_dim=len(bundle["gene_vocab"]), config=config, gene_vocab=bundle["gene_vocab"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def technical_embedding(model: torch.nn.Module, categorical: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    """Technical branch embedding before the gene-level projection.

    Mirrors ``ModularCodebookVQVAE.nuisance_effect`` including the ``<unknown>``
    exclusion, so an unseen OOD dataset contributes exactly zero.
    """
    rank = int(getattr(model, "nuisance_rank", 0) or 0)
    fields = tuple(getattr(model, "nuisance_fields", ()) or ())
    if rank <= 0 or not fields or not categorical:
        return torch.zeros(batch_size, max(rank, 1), device=device)
    embedding = torch.zeros(batch_size, rank, device=device)
    for name in fields:
        if name not in categorical:
            continue
        weight = model.nuisance_embeddings[name].weight
        categories = model.nuisance_categories[name]
        has_unknown = bool(categories) and str(categories[0]) == "<unknown>"
        if getattr(model, "exclude_unknown_from_nuisance_centering", False) and has_unknown and weight.shape[0] > 1:
            centered = torch.cat(
                [torch.zeros_like(weight[:1]), weight[1:] - weight[1:].mean(dim=0, keepdim=True)], dim=0
            )
        else:
            centered = weight - weight.mean(dim=0, keepdim=True)
        embedding = embedding + torch.nn.functional.embedding(categorical[name], centered)
    return embedding


def _splits_for(bundle: Dict[str, Any], splits: Sequence[str]) -> List[str]:
    available = [name for name in splits if name in bundle["splits"]]
    if not available:
        raise ValueError(f"bundle has none of the requested splits {list(splits)}")
    return available


@torch.no_grad()
def extract_representations(
    model: torch.nn.Module,
    config: Dict[str, Any],
    bundle: Dict[str, Any],
    device: torch.device,
    *,
    splits: Sequence[str] = DEFAULT_SPLITS,
    max_cells: int = 100_000,
    projection_dim: int = 256,
    seed: int = 1701,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> RepresentationPack:
    ensure_on_path()
    from torch.utils.data import ConcatDataset, Subset

    from vq_pipeline.bundle import warm_csr_caches
    from vq_pipeline.data import H5CSRDataset, make_dataloader
    from vq_pipeline.runtime import batch_to_device, categorical_indices_for_model, forward_batch

    split_names = _splits_for(bundle, splits)
    warm_csr_caches(bundle, config, split_names=split_names)
    datasets = [
        H5CSRDataset(bundle["splits"][name], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
        for name in split_names
    ]
    try:
        combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        total = len(combined)
        if max_cells and total > int(max_cells):
            rng = np.random.default_rng(seed)
            # Sampling is set valued; sorted indices keep reads file local.
            chosen = np.sort(rng.choice(total, size=int(max_cells), replace=False)).tolist()
            combined = Subset(combined, chosen)
        loader = make_dataloader(
            combined,
            int(batch_size or config["training"]["batch_size"]),
            False,
            int(config["training"]["num_workers"] if num_workers is None else num_workers),
        )

        projector = _Projector(projection_dim, seed, device)
        temperature = float(getattr(model, "conditional_code_usage_temperature", 0.5) or 0.5)
        buffers: Dict[str, List[np.ndarray]] = {}
        labels: Dict[str, List[Any]] = {name: [] for name in LABEL_FIELDS}
        code_indices: List[np.ndarray] = []
        soft_usage: List[np.ndarray] = []

        def store(name: str, tensor: torch.Tensor) -> None:
            buffers.setdefault(name, []).append(
                projector(name, tensor.float()).to(torch.float32).cpu().numpy()
            )

        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = forward_batch(model, batch, device)
            categorical = categorical_indices_for_model(batch, model, device)
            rows = int(batch["x"].shape[0])
            z_e = outputs["z_e"]
            z_q, indices, _, _, distance, _ = model.quantize(z_e)

            store("input_expression", batch["x"])
            store("encoder_z_e", z_e.reshape(rows, -1))
            store("bio_z_q", z_q.reshape(rows, -1))
            store("bio_reconstruction", outputs["biological_recon"])
            store("full_reconstruction", outputs["x_recon"])
            store("technical_embedding", technical_embedding(model, categorical, rows, device))
            onehot = torch.nn.functional.one_hot(indices, num_classes=int(model.codebook_size)).float()
            store("bio_code_onehot", onehot.reshape(rows, -1))

            code_indices.append(indices.cpu().numpy().astype(np.int16))
            soft_usage.append(
                torch.softmax(-distance / temperature, dim=-1).to(torch.float16).cpu().numpy()
            )
            for name in LABEL_FIELDS:
                if name in batch:
                    labels[name].extend(list(batch[name]))
    finally:
        for dataset in datasets:
            dataset.close()

    representations = {name: np.concatenate(chunks, axis=0) for name, chunks in buffers.items()}
    return RepresentationPack(
        representations=representations,
        labels={name: np.asarray(values, dtype=object) for name, values in labels.items() if values},
        code_indices=np.concatenate(code_indices, axis=0).astype(np.int64),
        soft_code_usage=np.concatenate(soft_usage, axis=0),
        meta={
            "splits": split_names,
            "num_cells": int(sum(chunk.shape[0] for chunk in code_indices)),
            "projection_dim": int(projection_dim),
            "code_usage_temperature": temperature,
            "num_axes": int(model.num_axes),
            "codebook_size": int(model.codebook_size),
        },
    )


__all__ = [
    "LABEL_FIELDS",
    "RepresentationPack",
    "build_model_for_bundle",
    "extract_representations",
    "load_atlas_model",
    "load_bundle",
    "technical_embedding",
]
