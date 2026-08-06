import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cellxgene_data, create_dataloaders
from vqvae_batch import VAE


def compute_lisi(z, labels, n_neighbors=30):
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(z)
    _, indices = nn.kneighbors(z)
    scores = []
    for i in range(len(z)):
        neighbor_labels = labels[indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        p = counts / n_neighbors
        scores.append(1.0 / np.sum(p ** 2))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--batch-key", default="donor_id")
    parser.add_argument("--celltype-key", default="cell_type")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-cells-per-batch", type=int, default=50)
    parser.add_argument("--min-cells-per-type", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=15)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="output/working")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    X, batches, cell_types, meta = load_cellxgene_data(
        args.data_path,
        batch_key=args.batch_key,
        celltype_key=args.celltype_key,
        min_cells_per_batch=args.min_cells_per_batch,
        min_cells_per_type=args.min_cells_per_type,
        n_top_genes=args.n_top_genes,
        max_batches=args.max_batches,
        max_cell_types=20,
    )
    n_genes = meta["n_genes"]
    n_batches = meta["n_batches"]
    n_ct = meta["n_cell_types"]
    print(f"  {X.shape[0]} cells, {n_genes} genes, {n_batches} batches, {n_ct} cell types")

    train_loader, val_loader, _ = create_dataloaders(
        X, batches, cell_types, batch_size=args.batch_size
    )

    print("Training VAE (no adversary - clean latent space)...")
    model = VAE(
        n_genes=n_genes,
        n_batches=n_batches,
        n_cell_types=None,
        latent_dim=args.latent_dim,
        hidden_dim=256,
        use_adversary=False,
        kl_weight=0.001,
    ).to(device)

    best_ckpt = f"{args.output_dir}/best.pt"
    if args.skip_training and os.path.exists(best_ckpt):
        print(f"  Skipping training, loading {best_ckpt}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                bl = batch["batch"].to(device)
                out = model(x, bl)
                loss = out["recon_loss"] + 0.001 * out.get("kl_loss", torch.tensor(0.0))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    bl = batch["batch"].to(device)
                    out = model(x, bl)
                    val_loss += out["recon_loss"].item()
            val_loss /= len(val_loader)
            dt = time.time() - t0

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_ckpt)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{args.epochs}  val_recon={val_loss:.4f}  {dt:.1f}s")

    print("Extracting latent representations...")
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    model.eval()
    z_list, batch_list, ct_list = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            mu, _ = model.encoder(x)
            z_list.append(mu.cpu().numpy())
            batch_list.append(batch["batch"].numpy())
            ct_list.append(batch["cell_type"].numpy())
    z = np.concatenate(z_list)
    batch_labels = np.concatenate(batch_list)
    ct_labels = np.concatenate(ct_list)

    print(f"  Latent shape: {z.shape}")

    batch_lisi_raw = compute_lisi(z, batch_labels)
    ct_lisi_raw = compute_lisi(z, ct_labels)

    print(f"BEFORE Harmony:  batch_LISI={batch_lisi_raw:.3f}  celltype_LISI={ct_lisi_raw:.3f}")

    print("Applying Harmony batch correction...")
    import harmonypy
    meta_data = {"batch": batch_labels}
    ho = harmonypy.run_harmony(z, meta_data, "batch", max_iter_harmony=50)
    z_harmony = ho.Z_corr

    batch_lisi_harmony = compute_lisi(z_harmony, batch_labels)
    ct_lisi_harmony = compute_lisi(z_harmony, ct_labels)

    print(f"AFTER Harmony:   batch_LISI={batch_lisi_harmony:.3f}  celltype_LISI={ct_lisi_harmony:.3f}")

    n_batches_val = len(np.unique(batch_labels))
    random_batch_lisi = float(np.log(n_batches_val) / np.log(2))

    print(f"  Random baseline batch_LISI = {random_batch_lisi:.3f}")
    print(f"  Perfect mixing batch_LISI  = {n_batches_val:.3f}")

    print("Generating UMAP plots...")
    try:
        import scanpy as sc
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for suffix, emb in [("raw", z), ("harmony", z_harmony)]:
            adata = sc.AnnData(X=emb.astype(np.float32))
            adata.obs["batch"] = batch_labels.astype(str)
            adata.obs["cell_type"] = ct_labels.astype(str)
            sc.pp.neighbors(adata, use_rep="X")
            sc.tl.umap(adata)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sc.pl.umap(adata, color="batch", ax=axes[0], show=False, title=f"Batch ({suffix})")
            sc.pl.umap(adata, color="cell_type", ax=axes[1], show=False, title=f"Cell Type ({suffix})")
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/umap_{suffix}.png", dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"  Saved umap_{suffix}.png")
    except Exception as e:
        print(f"  UMAP failed: {e}")

    results = {
        "before_harmony": {"batch_lisi": batch_lisi_raw, "celltype_lisi": ct_lisi_raw},
        "after_harmony": {"batch_lisi": batch_lisi_harmony, "celltype_lisi": ct_lisi_harmony},
        "random_baseline_batch_lisi": random_batch_lisi,
        "perfect_mixing_batch_lisi": n_batches_val,
        "n_cells": len(z),
        "n_batches": n_batches,
        "n_cell_types": n_ct,
    }
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Before Harmony: batch_LISI={batch_lisi_raw:.3f}  celltype_LISI={ct_lisi_raw:.3f}")
    print(f"  After Harmony:  batch_LISI={batch_lisi_harmony:.3f}  celltype_LISI={ct_lisi_harmony:.3f}")
    print(f"  Random baseline: {random_batch_lisi:.3f}")
    print(f"  Perfect mixing:  {n_batches_val:.3f}")
    print(f"  Results saved to {args.output_dir}/results.json")
    print(f"  UMAPs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
