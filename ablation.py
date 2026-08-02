import argparse
import json
import os
import subprocess
import sys


def run(config, tag, device, out_dir):
    cmd = [
        sys.executable, "train.py",
        "--n-cells", str(config["n_cells"]),
        "--n-genes", str(config["n_genes"]),
        "--n-batches", str(config["n_batches"]),
        "--n-cell-types", str(config["n_cell_types"]),
        "--n-codes", str(config["n_codes"]),
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--device", device,
        "--output-dir", os.path.join(out_dir, tag),
    ]
    if config.get("use_adversary"):
        cmd.append("--use-adversary")
        cmd += ["--adversary-alpha", str(config.get("adversary_alpha", 1.0))]
        cmd += ["--alpha-ramp", str(config.get("alpha_ramp", 8))]
    if config.get("use_ema"):
        cmd += ["--use-ema", "--ema-decay", str(config.get("ema_decay", 0.99))]
    if config.get("restart_dead_codes"):
        cmd.append("--restart-dead-codes")
    if config.get("mmd_weight"):
        cmd += ["--mmd-weight", str(config["mmd_weight"])]
    if config.get("code_batch_weight"):
        cmd += ["--code-batch-weight", str(config["code_batch_weight"])]
    subprocess.run(cmd, check=True)

    eval_cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint", os.path.join(out_dir, tag, "best_model.pt"),
        "--n-cells", str(config["eval_cells"]),
        "--n-genes", str(config["n_genes"]),
        "--n-batches", str(config["n_batches"]),
        "--n-cell-types", str(config["n_cell_types"]),
        "--output", os.path.join(out_dir, tag, "eval.json"),
    ]
    subprocess.run(eval_cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="output/ablation")
    parser.add_argument("--n-cells", type=int, default=5000)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--n-cell-types", type=int, default=4)
    parser.add_argument("--n-codes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-cells", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "mps" if __import__("torch").backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    base = {
        "n_cells": args.n_cells, "n_genes": args.n_genes,
        "n_batches": args.n_batches, "n_cell_types": args.n_cell_types,
        "n_codes": args.n_codes, "epochs": args.epochs,
        "batch_size": args.batch_size, "eval_cells": args.eval_cells,
    }

    variants = [
        ("baseline", {**base, "use_adversary": False}),
        ("mmd_strong", {**base, "use_ema": True, "restart_dead_codes": True, "mmd_weight": 300}),
        ("code_batch", {**base, "use_ema": True, "restart_dead_codes": True, "code_batch_weight": 5.0}),
        ("adv+mmd+code", {**base, "use_adversary": True, "use_ema": True, "restart_dead_codes": True,
                          "alpha_ramp": 8, "mmd_weight": 300, "code_batch_weight": 5.0}),
    ]

    summary = []
    for tag, cfg in variants:
        print(f"\n=== Config: {tag} ===")
        run(cfg, tag, args.device, args.out_dir)
        with open(os.path.join(args.out_dir, tag, "eval.json")) as f:
            res = json.load(f)
        summary.append({
            "variant": tag,
            "batch_clf_acc": res["batch_classifier_accuracy"],
            "kbet": res["kbet_rejection_rate"],
            "ilisi": res["ilisi"],
            "bio_ari": res["bio_ari"],
            "bio_nmi": res["bio_nmi"],
            "ct_acc": res["cell_type_classifier_accuracy"],
            "cross_batch_acc": res["cross_batch_mean_accuracy"],
            "active_codes": res["codebook"]["active_codes"],
        })

    with open(os.path.join(args.out_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Comparison ===")
    print(f"{'variant':<14} {'batchAcc':>8} {'kBET':>6} {'iLISI':>7} {'bioARI':>7} {'bioNMI':>7} {'ctAcc':>7} {'xBatch':>7} {'codes':>6}")
    for row in summary:
        print(f"{row['variant']:<14} {row['batch_clf_acc']:8.3f} {row['kbet']:6.3f} "
              f"{row['ilisi']:7.2f} {row['bio_ari']:7.3f} {row['bio_nmi']:7.3f} "
              f"{row['ct_acc']:7.3f} {row['cross_batch_acc']:7.3f} {row['active_codes']:6d}")
    print(f"\nFull results: {args.out_dir}/comparison.json")


if __name__ == "__main__":
    main()