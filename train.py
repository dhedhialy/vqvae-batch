import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_synthetic_data, create_dataloaders
from vqvae_batch import VQVAE


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0
    total_recon = 0
    total_vq = 0
    total_adv = 0
    total_ct = 0

    for batch in loader:
        x = batch["x"].to(device)
        batch_labels = batch["batch"].to(device)
        cell_types = batch.get("cell_type")
        if cell_types is not None:
            cell_types = cell_types.to(device)

        optimizer.zero_grad()
        outputs = model(x, batch_labels, cell_types)
        loss = outputs["recon_loss"] + outputs["vq_loss"]

        total_recon += outputs["recon_loss"].item()
        total_vq += outputs["vq_loss"].item()

        if "adv_logits" in outputs:
            adv_loss = F.cross_entropy(outputs["adv_logits"], batch_labels)
            loss = loss + adv_loss
            total_adv += adv_loss.item()

        if "ct_logits" in outputs and cell_types is not None:
            ct_loss = F.cross_entropy(outputs["ct_logits"], cell_types)
            loss = loss + ct_loss
            total_ct += ct_loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "vq": total_vq / n,
        "adv": total_adv / n,
        "ct": total_ct / n,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_recon = 0
    all_indices = []
    all_batches = []
    all_cell_types = []

    for batch in loader:
        x = batch["x"].to(device)
        batch_labels = batch["batch"]
        outputs = model(x, batch["batch"].to(device), None)
        total_recon += outputs["recon_loss"].item()
        all_indices.append(outputs["encoding_indices"].cpu())
        all_batches.append(batch_labels)
        if "cell_type" in batch:
            all_cell_types.append(batch["cell_type"])

    indices = torch.cat(all_indices)
    batches = torch.cat(all_batches)
    total_codes = model.vq.n_codes
    unique_codes = len(torch.unique(indices))
    code_counts = torch.zeros(total_codes)
    for idx in indices.view(-1):
        code_counts[idx] += 1
    probs = code_counts / code_counts.sum()
    perplexity = torch.exp(-torch.sum(probs[probs > 0] * torch.log(probs[probs > 0])))

    return {
        "recon_loss": total_recon / len(loader),
        "active_codes": unique_codes,
        "total_codes": total_codes,
        "active_frac": unique_codes / total_codes,
        "perplexity": perplexity.item(),
        "code_counts": code_counts.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--n-cell-types", type=int, default=8)
    parser.add_argument("--n-codes", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subsample", type=float, default=None,
                        help="Fraction of data to use (e.g. 0.3 for 30%%)")
    parser.add_argument("--use-adversary", action="store_true",
                        help="Enable adversarial batch classifier")
    parser.add_argument("--adversary-alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Generating synthetic scRNA-seq data...")
    X, batches, cell_types = generate_synthetic_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_batches=args.n_batches,
        n_cell_types=args.n_cell_types,
    )
    print(f"  Cells: {X.shape[0]}, Genes: {X.shape[1]}")
    print(f"  Batches: {args.n_batches}, Cell types: {args.n_cell_types}")

    train_loader, val_loader, dataset = create_dataloaders(
        X, batches, cell_types,
        batch_size=args.batch_size,
        subsample=args.subsample,
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = VQVAE(
        n_genes=args.n_genes,
        n_batches=args.n_batches,
        n_cell_types=args.n_cell_types,
        n_codes=args.n_codes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        use_adversary=args.use_adversary,
        adversary_alpha=args.adversary_alpha,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    metrics = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        val_metrics = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        metrics["train"].append(train_metrics)
        metrics["val"].append(val_metrics)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"recon={train_metrics['recon']:.4f} "
            f"vq={train_metrics['vq']:.4f} | "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"codes={val_metrics['active_codes']}/{val_metrics['total_codes']} "
            f"ppl={val_metrics['perplexity']:.2f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["recon_loss"] < best_val_loss:
            best_val_loss = val_metrics["recon_loss"]
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
            with open(f"{args.output_dir}/best_epoch.txt", "w") as f:
                f.write(str(epoch))

    save_config = vars(args)
    save_config["n_params"] = n_params
    torch.save(
        {
            "args": save_config,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        },
        f"{args.output_dir}/checkpoint_final.pt",
    )

    config = vars(args)
    config["n_params"] = n_params
    config["best_val_recon_loss"] = best_val_loss
    config["final_active_codes"] = val_metrics["active_codes"]
    config["final_perplexity"] = val_metrics["perplexity"]
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Best val recon loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.output_dir}/")
    print(f"\nTo train on another device, copy the '{args.output_dir}/' directory")
    print(f"and run: python train.py --load {args.output_dir}/checkpoint_final.pt")


if __name__ == "__main__":
    main()
