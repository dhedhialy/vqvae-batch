"""Real-data VQ-VAE training on cellxgene counts.

Produces latent embeddings z and discrete codes, then saves both for scoring
with the same LISI / cross-batch metrics as the scVI / Harmony / Scanorama
baselines in ~/vqvae-batch/output/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_real import load_cellxgene_counts, create_dataloaders  # noqa: E402
from vqvae_batch import VQVAE, batch_kernel_mmd, code_batch_dependence  # noqa: E402


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    recon = 0.0
    idx_all, b_all, ct_all = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x, batch["batch"].to(device), None)
        recon += out["recon_loss"].item()
        idx_all.append(out["encoding_indices"].cpu())
        b_all.append(batch["batch"])
        ct_all.append(batch["cell_type"])
    idx = torch.cat(idx_all).view(-1)
    counts = torch.bincount(idx, minlength=model.vq.n_codes)
    active = int((counts > 0).sum())
    p = counts / counts.sum().clamp(min=1)
    ppl = torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item()
    return {
        "recon_loss": recon / len(loader), "active_codes": active,
        "perplexity": ppl, "indices": idx, "batches": torch.cat(b_all), "cell_types": torch.cat(ct_all),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--batch-key", default="donor_id")
    p.add_argument("--celltype-key", default="cell_type")
    p.add_argument("--min-cells-per-batch", type=int, default=50)
    p.add_argument("--min-cells-per-type", type=int, default=50)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--max-batches", type=int, default=15)
    p.add_argument("--max-cell-types", type=int, default=20)
    p.add_argument("--n-codes", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--commitment-cost", type=float, default=0.5)
    p.add_argument("--commitment-warmup", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--restart-dead-codes", action="store_true")
    p.add_argument("--mmd-weight", type=float, default=0.0)
    p.add_argument("--code-batch-weight", type=float, default=2.0)
    p.add_argument("--use-adversary", action="store_true")
    p.add_argument("--alpha-ramp", type=int, default=6)
    p.add_argument("--adversary-alpha", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="output/real_vqvae")
    p.add_argument("--max-cells", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    print("[1] Loading raw counts...")
    X, batches, cell_types, meta = load_cellxgene_counts(
        args.data_path, batch_key=args.batch_key, celltype_key=args.celltype_key,
        min_cells_per_batch=args.min_cells_per_batch,
        min_cells_per_type=args.min_cells_per_type,
        n_top_genes=args.n_top_genes, max_batches=args.max_batches,
        max_cell_types=args.max_cell_types,
    )
    if args.max_cells and X.shape[0] > args.max_cells:
        rng = np.random.RandomState(42)
        pick = rng.choice(X.shape[0], args.max_cells, replace=False)
        X, batches, cell_types = X[pick], batches[pick], cell_types[pick]
        print(f"  subsampled to {X.shape[0]} cells")
    print(f"  [done] {X.shape[0]} x {X.shape[1]}, {meta['n_batches']} batches, {meta['n_cell_types']} types")

    train_loader, val_loader, _ = create_dataloaders(
        X, batches, cell_types, batch_size=args.batch_size)
    print(f"  train {len(train_loader.dataset)}, val {len(val_loader.dataset)}")

    print("[2] Building model...")
    model = VQVAE(
        n_genes=X.shape[1], n_batches=meta["n_batches"],
        n_cell_types=meta["n_cell_types"],
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        n_codes=args.n_codes, commitment_cost=args.commitment_cost,
        use_adversary=args.use_adversary, adversary_alpha=args.adversary_alpha,
        use_ema=args.use_ema,
    ).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    tr_metrics = {"loss": 0.0}

    print("\n[3] Training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        lsum = rsum = vsum = 0.0
        warmup = min(1.0, epoch / max(args.commitment_warmup, 1))
        model.vq.commitment_cost = args.commitment_cost * warmup
        if model.use_adversary and args.alpha_ramp > 0:
            model.adversary_alpha = args.adversary_alpha * min(1.0, epoch / args.alpha_ramp)
        opt.zero_grad()
        steps = 0
        for i, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            bl = batch["batch"].to(device)
            out = model(x, bl, None)
            loss = (out["recon_loss"] + out["vq_loss"]) / args.grad_accum_steps
            if "adv_logits" in out:
                loss = loss + F.cross_entropy(out["adv_logits"], bl).div(args.grad_accum_steps)
            if args.mmd_weight > 0:
                loss = loss + (args.mmd_weight * batch_kernel_mmd(out["z"], bl)).div(args.grad_accum_steps)
            if args.code_batch_weight > 0:
                loss = loss + (args.code_batch_weight * code_batch_dependence(
                    out["encoding_indices"], bl, model.vq.n_codes)).div(args.grad_accum_steps)
            loss.backward()
            lsum += out["recon_loss"].item() + out["vq_loss"].item()
            steps += 1
            if (i + 1) % args.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        tr_metrics = eval_epoch(model, val_loader, device)
        print(f"  epoch {epoch:3d}/{args.epochs} | loss~{lsum/len(val_loader):.3f} "
              f"val_recon={tr_metrics['recon_loss']:.3f} codes={tr_metrics['active_codes']}/{args.n_codes} "
              f"ppl={tr_metrics['perplexity']:.1f} | {time.time()-t0:.0f}s")
        if tr_metrics["recon_loss"] < best:
            best = tr_metrics["recon_loss"]
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       f"{args.output_dir}/best.pt")
            with open(f"{args.output_dir}/best_epoch.txt", "w") as f:
                f.write(str(epoch))

    cfg = vars(args)
    cfg["n_params"] = sum(p.numel() for p in model.parameters())
    cfg["best_val_recon"] = best
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nDone. best={best:.3f} -> {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()