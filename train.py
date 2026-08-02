import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import generate_synthetic_data, create_dataloaders
from vqvae_batch import VQVAE, batch_kernel_mmd, code_batch_dependence


def train_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    total_loss = 0
    total_recon = 0
    total_vq = 0
    total_adv = 0
    total_ct = 0
    total_mmd = 0

    # ponytail: commitment cost warmup — scVI warms KL from 0->1, same for VQ commitment
    warmup_frac = min(1.0, epoch / max(args.commitment_warmup, 1))
    commitment_cost = args.commitment_cost * warmup_frac
    model.vq.commitment_cost = commitment_cost

# DANN-style adversarial ramp: alpha grows 0->target over the first epochs so
    # the encoder first learns reconstruction, then pushes batch out of z
    if model.use_adversary and args.alpha_ramp > 0:
        ramp_frac = min(1.0, epoch / args.alpha_ramp)
        model.adversary_alpha = args.adversary_alpha * ramp_frac

    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        batch_labels = batch["batch"].to(device)
        cell_types = batch.get("cell_type")
        if cell_types is not None:
            cell_types = cell_types.to(device)

        outputs = model(x, batch_labels, cell_types)
        loss = outputs["recon_loss"] + outputs["vq_loss"]

        total_recon += outputs["recon_loss"].item()
        total_vq += outputs["vq_loss"].item()

        if args.mmd_weight > 0:
            mmd_loss = batch_kernel_mmd(outputs["z"], batch_labels)
            loss = loss + args.mmd_weight * mmd_loss
            total_mmd += mmd_loss.item()

        if args.code_batch_weight > 0:
            cb = code_batch_dependence(outputs["encoding_indices"], batch_labels, model.vq.n_codes)
            loss = loss + args.code_batch_weight * cb

        if "adv_logits" in outputs:
            adv_loss = F.cross_entropy(outputs["adv_logits"], batch_labels)
            loss = loss + adv_loss * args.adv_weight
            total_adv += adv_loss.item()

        if "ct_logits" in outputs and cell_types is not None:
            ct_loss = F.cross_entropy(outputs["ct_logits"], cell_types)
            loss = loss + ct_loss * args.ct_weight
            total_ct += ct_loss.item()

        loss = loss / args.grad_accum_steps
        loss.backward()

        if (i + 1) % args.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * args.grad_accum_steps

    if len(loader) % args.grad_accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "vq": total_vq / n,
        "adv": total_adv / n,
        "ct": total_ct / n,
        "mmd": total_mmd / n,
        "commitment": commitment_cost,
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
    # Data
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--n-cell-types", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=None)
    # Model
    parser.add_argument("--n-codes", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--commitment-warmup", type=int, default=10,
                        help="Epochs to linearly warm up commitment cost from 0")
    parser.add_argument("--use-adversary", action="store_true")
    parser.add_argument("--adversary-alpha", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--ct-weight", type=float, default=1.0)
    parser.add_argument("--mmd-weight", type=float, default=0.0,
                        help="Statistical batch-invariance (MMD) penalty weight. scVI-style "
                             "alternative to the adversarial — more stable.")
    parser.add_argument("--code-batch-weight", type=float, default=0.0,
                        help="Batch-independence penalty on the CODE histogram (the interpretable "
                             "object) — forces no code to be batch-specific")
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA codebook update (scVI-style stable update)")
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--restart-dead-codes", action="store_true",
                        help="Re-init unused codes each epoch (Jukebox trick)")
    parser.add_argument("--restart-threshold", type=float, default=0.005,
                        help="Fraction of validation cells a code must cover or it counts as dead")
    parser.add_argument("--restart-dead-epochs", type=int, default=3,
                        help="Consecutive epochs a code must be dead before restart")
    # Adversarial ramp
    parser.add_argument("--use-adversarial", action="store_true",
                        help="Alias for --use-adversary (kept for clarity)")
    parser.add_argument("--alpha-ramp", type=int, default=10,
                        help="Epochs to linearly ramp adversarial alpha from 0 to target")
    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="Epochs to wait before reducing LR on plateau")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    # Other
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
    eff_bs = args.batch_size * args.grad_accum_steps
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print(f"  Effective batch size: {eff_bs} ({args.batch_size} * {args.grad_accum_steps})")

    use_adversary = args.use_adversary or args.use_adversarial
    model = VQVAE(
        n_genes=args.n_genes,
        n_batches=args.n_batches,
        n_cell_types=args.n_cell_types,
        n_codes=args.n_codes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        commitment_cost=args.commitment_cost,
        use_adversary=use_adversary,
        adversary_alpha=args.adversary_alpha,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience,
        factor=args.lr_factor, min_lr=args.min_lr,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        metrics["train"].append(train_metrics)
        metrics["val"].append(val_metrics)

        scheduler.step(val_metrics["recon_loss"])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"recon={train_metrics['recon']:.4f} "
            f"vq={train_metrics['vq']:.4f} | "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"codes={val_metrics['active_codes']}/{val_metrics['total_codes']} "
            f"ppl={val_metrics['perplexity']:.2f} | "
            f"{elapsed:.1f}s | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["recon_loss"] < best_val_loss:
            best_val_loss = val_metrics["recon_loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
            with open(f"{args.output_dir}/best_epoch.txt", "w") as f:
                f.write(str(epoch))
        else:
            epochs_no_improve += 1

        # ponytail: reinit only codes dead for N consecutive epochs (prevents
        # churn — a code used by <1% of cells isn't broken, one never used for
        # a while is). Persistent counter lives on model.vq.dead_epochs.
        if args.restart_dead_codes:
            code_counts = torch.tensor(val_metrics["code_counts"], device=device)
            frac_count = code_counts / code_counts.sum().clamp(min=1)
            dead_now = (frac_count < args.restart_threshold).long()
            model.vq.dead_epochs = torch.clamp(model.vq.dead_epochs + dead_now, max=args.restart_dead_epochs)
            dead_idx = (model.vq.dead_epochs >= args.restart_dead_epochs).nonzero().view(-1)
            if dead_idx.numel() > 0:
                sample = next(iter(train_loader))["x"].to(device)
                model.restart_dead_codes(sample[: min(len(sample), dead_idx.numel())], dead_idx[:256])
                model.vq.dead_epochs[dead_idx] = 0
                print(f"    Restarted {dead_idx.numel()} dead codes")

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            break

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
    config["effective_batch_size"] = eff_bs
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Best val recon loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
