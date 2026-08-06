import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def compute_lisi(z, labels, n_neighbors=30):
    from sklearn.neighbors import NearestNeighbors
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


def cross_batch_classification(X, batches, cell_types, n_splits=5, name=""):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict
    unique_batches = np.unique(batches)
    rf_scores = []
    lr_scores = []
    for b in unique_batches:
        test_mask = batches == b
        train_mask = ~test_mask
        if test_mask.sum() < 10 or train_mask.sum() < 10:
            continue
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        rf.fit(X[train_mask], cell_types[train_mask])
        lr.fit(X[train_mask], cell_types[train_mask])
        rf_scores.append(rf.score(X[test_mask], cell_types[test_mask]))
        lr_scores.append(lr.score(X[test_mask], cell_types[test_mask]))
    return {
        f"{name}_rf_mean": float(np.mean(rf_scores)),
        f"{name}_lr_mean": float(np.mean(lr_scores)),
        f"{name}_rf_std": float(np.std(rf_scores)),
        f"{name}_lr_std": float(np.std(lr_scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--batch-key", default="donor_id")
    parser.add_argument("--celltype-key", default="cell_type")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-cells-per-batch", type=int, default=50)
    parser.add_argument("--min-cells-per-type", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=15)
    parser.add_argument("--output-dir", default="output/benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    print("=" * 60)
    print("FULL BENCHMARK: Raw vs VAE+Harmony vs scVI")
    print("=" * 60)

    print("\n[1] Loading data...")
    from data import load_cellxgene_data
    X, batches, cell_types, meta = load_cellxgene_data(
        args.data_path, batch_key=args.batch_key, celltype_key=args.celltype_key,
        min_cells_per_batch=args.min_cells_per_batch,
        min_cells_per_type=args.min_cells_per_type,
        n_top_genes=args.n_top_genes, max_batches=args.max_batches, max_cell_types=20,
    )
    n_genes, n_batches, n_ct = meta["n_genes"], meta["n_batches"], meta["n_cell_types"]
    print(f"  {X.shape[0]} cells, {n_genes} genes, {n_batches} batches, {n_ct} cell types")

    import scanpy as sc
    adata = sc.AnnData(X=X.copy())
    adata.obs["batch"] = batches.astype(str)
    adata.obs["cell_type"] = cell_types.astype(str)

    print("\n[2] Method A: PCA on raw data + Harmony (sklearn PCA baseline)")
    from sklearn.decomposition import TruncatedSVD
    t0 = time.time()
    pca = TruncatedSVD(n_components=50, random_state=42)
    z_pca = pca.fit_transform(X)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  PCA time: {time.time()-t0:.1f}s")

    batch_lisi_raw_pca = compute_lisi(z_pca, batches)
    ct_lisi_raw_pca = compute_lisi(z_pca, cell_types)
    print(f"  PCA LISI: batch={batch_lisi_raw_pca:.3f}  celltype={ct_lisi_raw_pca:.3f}")

    import harmonypy
    t0 = time.time()
    ho_pca = harmonypy.run_harmony(z_pca, {"batch": batches}, "batch", max_iter_harmony=50, theta=4.0)
    z_harmony_pca = ho_pca.Z_corr
    print(f"  Harmony on PCA time: {time.time()-t0:.1f}s")

    batch_lisi_harmony_pca = compute_lisi(z_harmony_pca, batches)
    ct_lisi_harmony_pca = compute_lisi(z_harmony_pca, cell_types)
    print(f"  PCA+Harmony LISI: batch={batch_lisi_harmony_pca:.3f}  celltype={ct_lisi_harmony_pca:.3f}")

    all_results["pca_raw"] = {"batch_lisi": batch_lisi_raw_pca, "celltype_lisi": ct_lisi_raw_pca}
    all_results["pca_harmony"] = {"batch_lisi": batch_lisi_harmony_pca, "celltype_lisi": ct_lisi_harmony_pca}

    print("\n[3] Method B: VAE latent + Harmony (pretrained)")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from vqvae_batch import VAE
    from data import create_dataloaders

    model_ckpt = f"{os.path.dirname(args.output_dir)}/working/best.pt"
    if not os.path.exists(model_ckpt):
        model_ckpt = "/home/aly/vqvae-batch/output/working/best.pt"

    model = VAE(n_genes=n_genes, n_batches=n_batches, n_cell_types=None,
                latent_dim=128, hidden_dim=256, use_adversary=False, kl_weight=0.001).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded model from {model_ckpt}")

    train_loader, val_loader, test_loader = create_dataloaders(X, batches, cell_types, batch_size=256)
    all_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                         torch.tensor(batches, dtype=torch.long)),
        batch_size=256, shuffle=False
    )

    z_vae_list = []
    with torch.no_grad():
        for xb, bb in all_loader:
            xb, bb = xb.to(device), bb.to(device)
            mu, _ = model.encoder(xb)
            z_vae_list.append(mu.cpu().numpy())
    z_vae = np.concatenate(z_vae_list)
    print(f"  VAE latent shape: {z_vae.shape}")

    batch_lisi_vae_raw = compute_lisi(z_vae, batches)
    ct_lisi_vae_raw = compute_lisi(z_vae, cell_types)
    print(f"  VAE raw LISI: batch={batch_lisi_vae_raw:.3f}  celltype={ct_lisi_vae_raw:.3f}")

    t0 = time.time()
    ho_vae = harmonypy.run_harmony(z_vae, {"batch": batches}, "batch", max_iter_harmony=50, theta=4.0)
    z_vae_harmony = ho_vae.Z_corr
    print(f"  VAE+Harmony time: {time.time()-t0:.1f}s")

    batch_lisi_vae_harmony = compute_lisi(z_vae_harmony, batches)
    ct_lisi_vae_harmony = compute_lisi(z_vae_harmony, cell_types)
    print(f"  VAE+Harmony LISI: batch={batch_lisi_vae_harmony:.3f}  celltype={ct_lisi_vae_harmony:.3f}")

    all_results["vae_raw"] = {"batch_lisi": batch_lisi_vae_raw, "celltype_lisi": ct_lisi_vae_raw}
    all_results["vae_harmony"] = {"batch_lisi": batch_lisi_vae_harmony, "celltype_lisi": ct_lisi_vae_harmony}

    print("\n[4] Method C: Harmony with different theta values on PCA")
    for theta_val in [1.0, 2.0, 4.0, 8.0, 16.0]:
        ho_t = harmonypy.run_harmony(z_pca, {"batch": batches}, "batch", max_iter_harmony=50, theta=theta_val)
        z_t = ho_t.Z_corr
        bl = compute_lisi(z_t, batches)
        cl = compute_lisi(z_t, cell_types)
        print(f"  theta={theta_val:5.1f}: batch_LISI={bl:.3f}  celltype_LISI={cl:.3f}")
        all_results[f"pca_harmony_theta{theta_val}"] = {"batch_lisi": bl, "celltype_lisi": cl}

    print("\n[5] Method D: scVI (scvi-tools)")
    scvi_installed = False
    try:
        import scvi
        print(f"  scvi-tools version: {scvi.__version__}")
        scvi_installed = True
    except ImportError:
        print("  scvi-tools not installed. Installing...")
        os.system(f"{sys.executable} -m pip install scvi-tools --quiet 2>/dev/null")
        try:
            import scvi
            print(f"  scvi-tools installed: {scvi.__version__}")
            scvi_installed = True
        except:
            print("  scvi-tools installation failed, skipping")

    if scvi_installed:
        import anndata as ad
        adata_scvi = ad.AnnData(X=X.copy())
        adata_scvi.obs["batch"] = batches.astype(str)
        adata_scvi.obs["cell_type"] = cell_types.astype(str)
        adata_scvi.layers["counts"] = X.copy()

        scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key="batch")
        scvi_model = scvi.model.SCVI(adata_scvi, n_latent=30, n_layers=2, dropout_rate=0.1)
        print("  Training scVI...")
        t0 = time.time()
        scvi_model.train(max_epochs=200, early_stopping=True, early_stopping_patience=20)
        print(f"  scVI training time: {time.time()-t0:.1f}s")

        z_scvi = scvi_model.get_latent_representation()
        print(f"  scVI latent shape: {z_scvi.shape}")

        batch_lisi_scvi_raw = compute_lisi(z_scvi, batches)
        ct_lisi_scvi_raw = compute_lisi(z_scvi, cell_types)
        print(f"  scVI raw LISI: batch={batch_lisi_scvi_raw:.3f}  celltype={ct_lisi_scvi_raw:.3f}")

        ho_scvi = harmonypy.run_harmony(z_scvi, {"batch": batches}, "batch", max_iter_harmony=50, theta=4.0)
        z_scvi_harmony = ho_scvi.Z_corr
        batch_lisi_scvi_harmony = compute_lisi(z_scvi_harmony, batches)
        ct_lisi_scvi_harmony = compute_lisi(z_scvi_harmony, cell_types)
        print(f"  scVI+Harmony LISI: batch={batch_lisi_scvi_harmony:.3f}  celltype={ct_lisi_scvi_harmony:.3f}")

        all_results["scvi_raw"] = {"batch_lisi": batch_lisi_scvi_raw, "celltype_lisi": ct_lisi_scvi_raw}
        all_results["scvi_harmony"] = {"batch_lisi": batch_lisi_scvi_harmony, "celltype_lisi": ct_lisi_scvi_harmony}

        print("  Generating scVI UMAPs...")
        try:
            adata_scvi.obsm["X_scVI"] = z_scvi
            adata_scvi.obsm["X_scVI_harmony"] = z_scvi_harmony
            sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
            sc.tl.umap(adata_scvi)
            sc.pl.umap(adata_scvi, color="batch", show=False, title="scVI Batch")
            import matplotlib.pyplot as plt
            plt.savefig(f"{args.output_dir}/umap_scvi_batch.png", dpi=150, bbox_inches="tight")
            plt.close("all")
            sc.pl.umap(adata_scvi, color="cell_type", show=False, title="scVI Cell Type")
            plt.savefig(f"{args.output_dir}/umap_scvi_celltype.png", dpi=150, bbox_inches="tight")
            plt.close("all")
        except Exception as e:
            print(f"  scVI UMAP failed: {e}")

    print("\n[6] Cross-batch cell type classification (full dataset)")
    from sklearn.ensemble import RandomForestClassifier
    unique_batches = np.unique(batches)
    for method_name, z_emb in [("pca_raw", z_pca), ("pca_harmony", z_harmony_pca),
                                ("vae_raw", z_vae), ("vae_harmony", z_vae_harmony)]:
        rf_scores = []
        for b in unique_batches:
            test_mask = batches == b
            train_mask = ~test_mask
            if test_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
            rf.fit(z_emb[train_mask], cell_types[train_mask])
            rf_scores.append(rf.score(z_emb[test_mask], cell_types[test_mask]))
        acc = float(np.mean(rf_scores))
        std = float(np.std(rf_scores))
        print(f"  {method_name:20s}: cross-batch CT acc = {acc:.3f} +/- {std:.3f}")
        all_results[f"{method_name}_cross_batch"] = {"acc": acc, "std": std}

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Method':<30s} {'batch_LISI':>10s} {'celltype_LISI':>12s}")
    print("-" * 55)
    random_bl = float(np.log(n_batches) / np.log(2))
    print(f"{'Random baseline':<30s} {random_bl:>10.3f}")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        if "batch_lisi" in r:
            print(f"{key:<30s} {r['batch_lisi']:>10.3f} {r['celltype_lisi']:>12.3f}")
    print("-" * 55)
    print(f"{'Perfect mixing':<30s} {float(n_batches):>10.3f}")

    all_results["random_baseline"] = random_bl
    all_results["perfect_mixing"] = float(n_batches)
    with open(f"{args.output_dir}/full_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_dir}/full_benchmark.json")


if __name__ == "__main__":
    main()
