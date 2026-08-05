"""Evaluate the real-data VQ-VAE with the exact metrics of final_compare.py.

Feeds RAW counts (what the model was trained on) and produces a results.json with
batch_lisi / celltype_lisi / cross_batch_acc so it slots into the baseline table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_real import load_cellxgene_counts  # noqa: E402
from vqvae_batch import VQVAE  # noqa: E402


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
    scores = []
    for b in np.unique(batches):
        test_mask = batches == b
        train_mask = ~test_mask
        if test_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(z[train_mask], cell_types[train_mask])
        scores.append(rf.score(z[test_mask], cell_types[test_mask]))
    return float(np.mean(scores)), float(np.std(scores))


@torch.no_grad()
def get_latents_in_chunks(model, X, batch_size, device):
    model.eval()
    z_all, idx_all = [], []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.FloatTensor(X[i:i + batch_size]).to(device)
        zb, _, ib = model.get_latents(xb)
        z_all.append(zb.cpu())
        idx_all.append(ib.cpu())
    return torch.cat(z_all).numpy(), torch.cat(idx_all).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--batch-key", default="donor_id")
    ap.add_argument("--celltype-key", default="cell_type")
    ap.add_argument("--min-cells-per-batch", type=int, default=50)
    ap.add_argument("--min-cells-per-type", type=int, default=50)
    ap.add_argument("--n-top-genes", type=int, default=2000)
    ap.add_argument("--max-batches", type=int, default=15)
    ap.add_argument("--max-cell-types", type=int, default=20)
    ap.add_argument("--max-cells", type=int, default=None, help="cap for LISI speed")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="output/real_eval.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("Loading data (raw counts)...")
    X, batches, cell_types, meta = load_cellxgene_counts(
        args.data_path, batch_key=args.batch_key, celltype_key=args.celltype_key,
        min_cells_per_batch=args.min_cells_per_batch,
        min_cells_per_type=args.min_cells_per_type,
        n_top_genes=args.n_top_genes,
        max_batches=args.max_batches, max_cell_types=args.max_cell_types,
    )
    print(f"[done] {X.shape[0]} x {X.shape[1]}")

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    model = VQVAE(
        n_genes=X.shape[1],
        n_batches=int(sd["decoder.batch_embedding.weight"].shape[0]),
        n_cell_types=int(sd["classifier.net.2.weight"].shape[0]),
        n_codes=cfg.get("n_codes", 64),
        latent_dim=cfg.get("latent_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 256),
        commitment_cost=cfg.get("commitment_cost", 0.5),
        use_adversary=cfg.get("use_adversary", False),
        adversary_alpha=cfg.get("adversary_alpha", 1.0),
        use_ema=cfg.get("use_ema", False),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    t0 = time.time()
    z, indices = get_latents_in_chunks(model, X, args.batch_size, device)
    print(f"latents: {z.shape} in {time.time() - t0:.0f}s")

    if args.max_cells and z.shape[0] > args.max_cells:
        rng = np.random.RandomState(42)
        pick = rng.choice(z.shape[0], args.max_cells, replace=False)
        z, batches, cell_types = z[pick], batches[pick], cell_types[pick]

    print("LISI on continuous z (as baselines do)...")
    batch_lisi = compute_lisi(z, batches)
    celltype_lisi = compute_lisi(z, cell_types)
    print("cross-batch cell-type transfer...")
    xb_acc, xb_std = cross_batch_ct_acc(z, batches, cell_types)

    results = {
        "model": "vqvae_real_raw_counts",
        "batch_lisi": batch_lisi,
        "celltype_lisi": celltype_lisi,
        "cross_batch_acc": xb_acc,
        "cross_batch_std": xb_std,
        "random_baseline_batch_lisi": float(np.log(len(np.unique(batches))) / np.log(2)),
        "perfect_mixing_batch_lisi": float(len(np.unique(batches))),
        "n_cells": z.shape[0],
        "n_batches": int(batches.max()) + 1,
        "n_cell_types": int(cell_types.max()) + 1,
        "training_config": cfg,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()