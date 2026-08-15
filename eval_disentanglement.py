"""Disentanglement evaluation for VQ-VAE checkpoints (ID + OOD).

This script extends the project's existing real-data evaluation with:
1) multi-factor leakage probes (e.g. dataset, assay, donor),
2) biology-retention probes and leave-one-group transfer,
3) VQ-specific code-usage alignment metrics,
4) optional OOD transfer from in-distribution probes.

It is designed to be practical on shared servers: use --max-cells-per-split and
small probe models to avoid excessive resource usage.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vqvae_batch import VQVAE


@dataclass
class LoadedSplit:
    name: str
    x: np.ndarray
    obs: Dict[str, np.ndarray]
    genes: List[str]


def _as_dense(matrix):
    from scipy.sparse import issparse

    return np.asarray(matrix.toarray() if issparse(matrix) else matrix, dtype=np.float32)


def _parse_csv_fields(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _counts_and_genes(adata):
    if adata.raw is not None:
        return _as_dense(adata.raw.X), np.asarray(adata.raw.var_names, dtype=str)
    if "counts" in adata.layers:
        return _as_dense(adata.layers["counts"]), np.asarray(adata.var_names, dtype=str)
    return _as_dense(adata.X), np.asarray(adata.var_names, dtype=str)


def _select_hvg_indices(counts: np.ndarray, n_top: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lib = counts.sum(axis=1, keepdims=True)
        log_norm = np.log1p(counts / np.maximum(lib, 1.0) * 1e4)
    n_expr = (log_norm > 0).sum(axis=0)
    keep = n_expr >= 10
    candidate_idx = np.where(keep)[0]
    if candidate_idx.size == 0:
        candidate_idx = np.arange(counts.shape[1])
    vals = log_norm[:, candidate_idx]
    disp = (vals.var(axis=0) - vals.mean(axis=0)) / np.maximum(vals.mean(axis=0), 1e-9)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    picked = candidate_idx[np.argsort(disp)[::-1][:n_top]]
    return np.sort(picked)


def _build_feature_genes(id_path: str, expected_n_genes: int) -> List[str]:
    import anndata as ad

    adata = ad.read_h5ad(id_path)
    counts, genes = _counts_and_genes(adata)
    if counts.shape[1] == expected_n_genes:
        return genes.tolist()
    if counts.shape[1] < expected_n_genes:
        raise ValueError(
            f"{id_path} has only {counts.shape[1]} genes, but checkpoint expects {expected_n_genes}."
        )
    hvg_idx = _select_hvg_indices(counts, expected_n_genes)
    return genes[hvg_idx].tolist()


def _load_split(
    split_name: str,
    h5ad_path: str,
    fields: Sequence[str],
    feature_genes: Sequence[str],
    max_cells: Optional[int],
    seed: int,
) -> LoadedSplit:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    counts, genes = _counts_and_genes(adata)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    missing = [g for g in feature_genes if g not in gene_to_idx]
    if missing:
        raise ValueError(
            f"{split_name} is missing {len(missing)} required genes. "
            "Use harmonized datasets with shared gene vocabulary."
        )

    col_idx = np.asarray([gene_to_idx[g] for g in feature_genes], dtype=np.int64)
    x = counts[:, col_idx]

    valid = np.isfinite(x).all(axis=1)
    x = x[valid]
    if x.shape[0] == 0:
        raise ValueError(f"{split_name} has no valid cells after finite-value filtering.")

    obs: Dict[str, np.ndarray] = {}
    for field in fields:
        if field not in adata.obs.columns:
            raise ValueError(f"Field '{field}' not found in {split_name}: {h5ad_path}")
        vals = adata.obs[field].astype(str).to_numpy()[valid]
        obs[field] = vals

    if max_cells is not None and x.shape[0] > max_cells:
        rng = np.random.RandomState(seed)
        pick = rng.choice(x.shape[0], max_cells, replace=False)
        x = x[pick]
        for field in obs:
            obs[field] = obs[field][pick]

    return LoadedSplit(name=split_name, x=x, obs=obs, genes=list(feature_genes))


@torch.no_grad()
def _encode_latents(
    model: VQVAE, x: np.ndarray, batch_size: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    z_all: List[torch.Tensor] = []
    idx_all: List[torch.Tensor] = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.as_tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
        z, _, idx = model.get_latents(xb)
        z_all.append(z.cpu())
        idx_all.append(idx.cpu())
    return torch.cat(z_all).numpy(), torch.cat(idx_all).numpy()


def _label_encode(series: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    uniq = np.unique(series)
    to_i = {v: i for i, v in enumerate(uniq)}
    return np.asarray([to_i[v] for v in series], dtype=np.int64), uniq.tolist()


def _compute_lisi(repr_array: np.ndarray, labels: np.ndarray, n_neighbors: int = 30) -> float:
    from sklearn.neighbors import NearestNeighbors

    if len(repr_array) < 3:
        return 1.0
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(repr_array) - 1), n_jobs=-1)
    nn.fit(repr_array)
    _, indices = nn.kneighbors(repr_array)
    scores = []
    for i in range(len(repr_array)):
        neighbor_labels = labels[indices[i]]
        _, counts = np.unique(neighbor_labels, return_counts=True)
        p = counts / counts.sum()
        scores.append(1.0 / np.sum(p**2))
    return float(np.mean(scores))


def _probe_scores(x: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, float]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    if len(np.unique(y)) < 2:
        return {}
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    out: Dict[str, float] = {}

    lr = LogisticRegression(max_iter=200)
    lr.fit(x_train, y_train)
    pred_lr = lr.predict(x_test)
    out["logreg_acc"] = float(accuracy_score(y_test, pred_lr))
    out["logreg_macro_f1"] = float(f1_score(y_test, pred_lr, average="macro"))

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1)
    rf.fit(x_train, y_train)
    pred_rf = rf.predict(x_test)
    out["rf_acc"] = float(accuracy_score(y_test, pred_rf))
    out["rf_macro_f1"] = float(f1_score(y_test, pred_rf, average="macro"))
    out["chance_acc"] = float(1.0 / len(np.unique(y)))
    return out


def _leave_one_group_transfer(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42
) -> Dict[str, float]:
    from sklearn.ensemble import RandomForestClassifier

    vals = np.unique(groups)
    scores: List[float] = []
    for g in vals:
        test_mask = groups == g
        train_mask = ~test_mask
        if test_mask.sum() < 20 or train_mask.sum() < 20:
            continue
        if len(np.unique(y[train_mask])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=seed, n_jobs=-1)
        clf.fit(x[train_mask], y[train_mask])
        scores.append(float(clf.score(x[test_mask], y[test_mask])))
    if not scores:
        return {}
    return {"mean_acc": float(np.mean(scores)), "std_acc": float(np.std(scores)), "n_groups": int(len(scores))}


def _fit_probe_train_eval_test(
    x_train: np.ndarray, y_train_raw: np.ndarray, x_test: np.ndarray, y_test_raw: np.ndarray, seed: int = 42
) -> Dict[str, float]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    common = sorted(set(np.unique(y_train_raw)).intersection(set(np.unique(y_test_raw))))
    if len(common) < 2:
        return {}
    keep_train = np.isin(y_train_raw, common)
    keep_test = np.isin(y_test_raw, common)
    if keep_train.sum() < 50 or keep_test.sum() < 50:
        return {}

    to_i = {v: i for i, v in enumerate(common)}
    y_train = np.asarray([to_i[v] for v in y_train_raw[keep_train]], dtype=np.int64)
    y_test = np.asarray([to_i[v] for v in y_test_raw[keep_test]], dtype=np.int64)
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=seed, n_jobs=-1)
    clf.fit(x_train[keep_train], y_train)
    pred = clf.predict(x_test[keep_test])
    return {
        "rf_acc": float(accuracy_score(y_test, pred)),
        "rf_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "n_common_classes": int(len(common)),
        "n_test_cells_used": int(keep_test.sum()),
    }


def _conditional_code_tv(
    code_idx: np.ndarray,
    target: np.ndarray,
    context: np.ndarray,
    n_codes: int,
    min_cells: int,
) -> Dict[str, object]:
    contexts = np.unique(context)
    per_ctx = {}
    tvs = []
    for ctx in contexts:
        m = context == ctx
        if m.sum() < min_cells:
            continue
        tvals = np.unique(target[m])
        if len(tvals) < 2:
            continue
        dists = []
        hist_list = []
        for tv in tvals:
            tm = m & (target == tv)
            if tm.sum() < min_cells:
                continue
            h = np.bincount(code_idx[tm], minlength=n_codes).astype(np.float64)
            h /= max(h.sum(), 1.0)
            hist_list.append(h)
        if len(hist_list) < 2:
            continue
        mean_h = np.mean(hist_list, axis=0)
        for h in hist_list:
            dists.append(0.5 * np.abs(h - mean_h).sum())
        per_ctx[str(ctx)] = float(np.mean(dists))
        tvs.extend(dists)
    if not tvs:
        return {"mean_tv": None, "n_contexts": 0, "per_context": {}}
    return {"mean_tv": float(np.mean(tvs)), "n_contexts": int(len(per_ctx)), "per_context": per_ctx}


def _evaluate_split(
    split: LoadedSplit,
    z: np.ndarray,
    code_idx: np.ndarray,
    batch_fields: Sequence[str],
    bio_field: str,
    transfer_group_field: Optional[str],
    context_field: Optional[str],
    min_cells: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "n_cells": int(z.shape[0]),
        "n_features": int(z.shape[1]),
        "n_codes": int(np.unique(code_idx).size),
    }

    bio_y, bio_classes = _label_encode(split.obs[bio_field])
    out["bio_field"] = bio_field
    out["bio_n_classes"] = int(len(bio_classes))
    out["bio_probe"] = _probe_scores(z, bio_y)
    out["bio_lisi"] = _compute_lisi(z, bio_y)

    if transfer_group_field is not None:
        if transfer_group_field not in split.obs:
            raise ValueError(f"transfer_group_field '{transfer_group_field}' missing in split '{split.name}'")
        grp_y, _ = _label_encode(split.obs[transfer_group_field])
        out["bio_transfer_lodo"] = _leave_one_group_transfer(z, bio_y, grp_y)

    out["batch_metrics"] = {}
    for field in batch_fields:
        y, classes = _label_encode(split.obs[field])
        out["batch_metrics"][field] = {
            "n_classes": int(len(classes)),
            "probe": _probe_scores(z, y),
            "lisi": _compute_lisi(z, y),
        }

    if context_field is not None:
        if context_field not in split.obs:
            raise ValueError(f"context_field '{context_field}' missing in split '{split.name}'")
        if len(batch_fields) == 0:
            out["conditional_code_usage"] = None
        else:
            out["conditional_code_usage"] = _conditional_code_tv(
                code_idx=code_idx,
                target=split.obs[batch_fields[0]],
                context=split.obs[context_field],
                n_codes=int(np.max(code_idx) + 1),
                min_cells=min_cells,
            )
    return out


def _load_model(checkpoint_path: str, n_genes: int, device: torch.device) -> VQVAE:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    cfg_path = os.path.join(ckpt_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    n_batches = int(sd["decoder.batch_embedding.weight"].shape[0])
    n_cell_types = None
    if "classifier.net.2.weight" in sd:
        n_cell_types = int(sd["classifier.net.2.weight"].shape[0])

    model = VQVAE(
        n_genes=n_genes,
        n_batches=n_batches,
        n_cell_types=n_cell_types,
        n_codes=cfg.get("n_codes", 64),
        latent_dim=cfg.get("latent_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 256),
        commitment_cost=cfg.get("commitment_cost", 0.5),
        use_adversary=cfg.get("use_adversary", False),
        adversary_alpha=cfg.get("adversary_alpha", 1.0),
        use_ema=cfg.get("use_ema", False),
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--id-data-path", required=True)
    ap.add_argument(
        "--ood-data-paths",
        default="",
        help="Comma-separated list of OOD h5ad paths (optional).",
    )
    ap.add_argument(
        "--ood-names",
        default="",
        help="Comma-separated names for OOD splits. Must match --ood-data-paths length if provided.",
    )
    ap.add_argument("--batch-fields", default="donor_id")
    ap.add_argument("--bio-field", default="cell_type")
    ap.add_argument(
        "--transfer-group-field",
        default="",
        help="Field for leave-one-group transfer (e.g. donor_id or dataset_id).",
    )
    ap.add_argument(
        "--context-field",
        default="",
        help="Context field for conditional code-usage alignment (e.g. coarse_cell_type).",
    )
    ap.add_argument("--conditional-min-cells", type=int, default=8)
    ap.add_argument("--max-cells-per-split", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="output/disentanglement_eval.json")
    args = ap.parse_args()

    batch_fields = _parse_csv_fields(args.batch_fields)
    if not batch_fields:
        raise ValueError("At least one batch field is required in --batch-fields.")
    transfer_group_field = args.transfer_group_field.strip() or None
    context_field = args.context_field.strip() or None

    ood_paths = _parse_csv_fields(args.ood_data_paths)
    ood_names = _parse_csv_fields(args.ood_names)
    if ood_paths and ood_names and len(ood_paths) != len(ood_names):
        raise ValueError("If --ood-names is set, it must match --ood-data-paths in length.")
    if ood_paths and not ood_names:
        ood_names = [f"ood_{i+1}" for i in range(len(ood_paths))]

    device = torch.device(args.device)

    # Build the gene panel from ID split so all OOD splits are projected on the same space.
    model_tmp_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_tmp_sd = model_tmp_ckpt.get("model", model_tmp_ckpt)
    expected_n_genes = int(model_tmp_sd["encoder.net.0.weight"].shape[1])
    feature_genes = _build_feature_genes(args.id_data_path, expected_n_genes)

    all_fields = sorted(set(batch_fields + [args.bio_field] + ([transfer_group_field] if transfer_group_field else []) + ([context_field] if context_field else [])))
    id_split = _load_split(
        split_name="id",
        h5ad_path=args.id_data_path,
        fields=all_fields,
        feature_genes=feature_genes,
        max_cells=args.max_cells_per_split,
        seed=args.seed,
    )

    model = _load_model(args.checkpoint, n_genes=len(feature_genes), device=device)
    id_z, id_idx = _encode_latents(model, id_split.x, args.batch_size, device)

    results: Dict[str, object] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "feature_genes_n": len(feature_genes),
        "batch_fields": batch_fields,
        "bio_field": args.bio_field,
        "transfer_group_field": transfer_group_field,
        "context_field": context_field,
        "splits": {},
        "ood_transfer_from_id": {},
    }

    results["splits"]["id"] = _evaluate_split(
        split=id_split,
        z=id_z,
        code_idx=id_idx,
        batch_fields=batch_fields,
        bio_field=args.bio_field,
        transfer_group_field=transfer_group_field,
        context_field=context_field,
        min_cells=args.conditional_min_cells,
    )

    for name, path in zip(ood_names, ood_paths):
        split = _load_split(
            split_name=name,
            h5ad_path=path,
            fields=all_fields,
            feature_genes=feature_genes,
            max_cells=args.max_cells_per_split,
            seed=args.seed,
        )
        z, idx = _encode_latents(model, split.x, args.batch_size, device)
        results["splits"][name] = _evaluate_split(
            split=split,
            z=z,
            code_idx=idx,
            batch_fields=batch_fields,
            bio_field=args.bio_field,
            transfer_group_field=transfer_group_field,
            context_field=context_field,
            min_cells=args.conditional_min_cells,
        )

        # OOD transfer: train on ID and test on OOD for key targets with overlapping classes.
        results["ood_transfer_from_id"][name] = {}
        for field in [args.bio_field] + list(batch_fields):
            scores = _fit_probe_train_eval_test(
                x_train=id_z,
                y_train_raw=id_split.obs[field],
                x_test=z,
                y_test_raw=split.obs[field],
                seed=args.seed,
            )
            results["ood_transfer_from_id"][name][field] = scores

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
