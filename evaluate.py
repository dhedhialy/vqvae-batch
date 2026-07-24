import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from data import generate_synthetic_data, create_dataloaders, SingleCellDataset
from vqvae_batch import VQVAE


@torch.no_grad()
def get_representations(model, loader, device):
    model.eval()
    z_list, z_q_list, idx_list, batch_list, ct_list = [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        z, z_q, idx = model.get_latents(x)
        z_list.append(z.cpu())
        z_q_list.append(z_q.cpu())
        idx_list.append(idx.cpu())
        batch_list.append(batch["batch"])
        if "cell_type" in batch:
            ct_list.append(batch["cell_type"])
    out = {
        "z": torch.cat(z_list).numpy(),
        "z_q": torch.cat(z_q_list).numpy(),
        "indices": torch.cat(idx_list).numpy(),
        "batch": torch.cat(batch_list).numpy(),
    }
    if ct_list:
        out["cell_type"] = torch.cat(ct_list).numpy()
    return out


def batch_classification_score(z, batch_labels):
    le = LabelEncoder()
    y = le.fit_transform(batch_labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(z, y)
    return accuracy_score(y, clf.predict(z))


def cell_type_classification_score(z, cell_types):
    le = LabelEncoder()
    y = le.fit_transform(cell_types)
    n_classes = len(le.classes_)
    if n_classes < 2:
        return 0.0, 0.0
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(z, y)
    preds = clf.predict(z)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    return acc, f1


def cross_batch_classification(z, batch_labels, cell_types):
    le_ct = LabelEncoder()
    y = le_ct.fit_transform(cell_types)
    unique_batches = np.unique(batch_labels)
    scores = {}
    for held_out in unique_batches:
        train_mask = batch_labels != held_out
        test_mask = batch_labels == held_out
        if train_mask.sum() < 10 or test_mask.sum() < 10:
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(z[train_mask], y[train_mask])
        preds = clf.predict(z[test_mask])
        scores[int(held_out)] = {
            "accuracy": float(accuracy_score(y[test_mask], preds)),
            "f1": float(f1_score(y[test_mask], preds, average="macro")),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }
    return scores


def codebook_analysis(indices, batch_labels, cell_types, n_codes):
    n_batches = len(np.unique(batch_labels))
    n_cts = len(np.unique(cell_types))
    code_by_batch = np.zeros((n_codes, n_batches))
    code_by_ct = np.zeros((n_codes, n_cts))
    for i in range(len(indices)):
        code_by_batch[indices[i], batch_labels[i]] += 1
        code_by_ct[indices[i], cell_types[i]] += 1
    code_by_batch = code_by_batch / code_by_batch.sum(axis=1, keepdims=True).clip(min=1)
    code_by_ct = code_by_ct / code_by_ct.sum(axis=1, keepdims=True).clip(min=1)
    usage = np.bincount(indices, minlength=n_codes)
    active = (usage > 0).sum()
    probs = usage / usage.sum()
    perplexity = np.exp(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    return {
        "active_codes": int(active),
        "total_codes": n_codes,
        "active_fraction": float(active / n_codes),
        "perplexity": float(perplexity),
        "code_usage": usage.tolist(),
        "code_by_batch": code_by_batch.tolist(),
        "code_by_cell_type": code_by_ct.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-cells", type=int, default=2000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--n-cell-types", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="output/eval_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "args" in checkpoint:
        model_args = checkpoint["args"]
    else:
        with open("output/config.json") as f:
            model_args = json.load(f)

    model = VQVAE(
        n_genes=model_args.get("n_genes", args.n_genes),
        n_batches=model_args.get("n_batches", args.n_batches),
        n_cell_types=model_args.get("n_cell_types", args.n_cell_types),
        n_codes=model_args.get("n_codes", 64),
        latent_dim=model_args.get("latent_dim", 128),
        hidden_dim=model_args.get("hidden_dim", 256),
        use_adversary=model_args.get("use_adversary", False),
        adversary_alpha=model_args.get("adversary_alpha", 1.0),
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded.")

    print("Generating evaluation data...")
    X, batches, cell_types = generate_synthetic_data(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_batches=args.n_batches,
        n_cell_types=args.n_cell_types,
        seed=99,
    )
    _, loader, _ = create_dataloaders(
        X, batches, cell_types, batch_size=args.batch_size
    )

    print("Computing representations...")
    reps = get_representations(model, loader, device)
    print(f"  z shape: {reps['z'].shape}")
    print(f"  Unique indices: {len(np.unique(reps['indices']))}")

    results = {}

    print("Batch classification score...")
    batch_acc = batch_classification_score(reps["z"], reps["batch"])
    results["batch_classifier_accuracy"] = float(batch_acc)
    print(f"  Batch classifier acc: {batch_acc:.4f}")

    print("Cell type classification score...")
    ct_acc, ct_f1 = cell_type_classification_score(reps["z"], reps["cell_type"])
    results["cell_type_classifier_accuracy"] = float(ct_acc)
    results["cell_type_classifier_f1"] = float(ct_f1)
    print(f"  Cell type classifier acc: {ct_acc:.4f}, f1: {ct_f1:.4f}")

    print("Cross-batch classification...")
    cross_batch = cross_batch_classification(reps["z"], reps["batch"], reps["cell_type"])
    results["cross_batch_classification"] = cross_batch
    mean_cross_acc = np.mean([v["accuracy"] for v in cross_batch.values()])
    results["cross_batch_mean_accuracy"] = float(mean_cross_acc)
    print(f"  Mean cross-batch acc: {mean_cross_acc:.4f}")

    print("Codebook analysis...")
    code_analysis = codebook_analysis(
        reps["indices"], reps["batch"], reps["cell_type"],
        model.vq.n_codes,
    )
    results["codebook"] = code_analysis
    print(f"  Active codes: {code_analysis['active_codes']}/{code_analysis['total_codes']}")
    print(f"  Perplexity: {code_analysis['perplexity']:.2f}")

    results["config"] = vars(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
