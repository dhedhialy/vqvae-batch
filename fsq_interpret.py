import os, sys, json
import numpy as np
from scipy.stats import hypergeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT = "/home/aly/vqvae-batch/output/fsq"
RUN = "L3x8_adv1.0"

print("=" * 65, flush=True)
print("Interpretability + pathway validation for FSQ-VAE", flush=True)
print("=" * 65, flush=True)

print("\n[1] Loading data + codes...", flush=True)
from data import load_cellxgene_data
X, batches, cell_types, meta = load_cellxgene_data(
    "/data/bhy/czcellxgene/h5ads/0b75c598-0893-4216-afe8-5414cab7739d.h5ad",
    batch_key="donor_id", celltype_key="cell_type",
    min_cells_per_batch=50, min_cells_per_type=50,
    n_top_genes=2000, max_batches=15, max_cell_types=20,
)
ct_names = sorted(np.unique(cell_types))
hvg_names = np.array(meta["gene_names"])
print(f"  genes: {len(hvg_names)}, first: {hvg_names[:3].tolist()}", flush=True)

codes = np.load(f"{OUTPUT}/codes_{RUN}.npy")
code_id = np.load(f"{OUTPUT}/code_id_{RUN}.npy").ravel()
print(f"  codes shape: {codes.shape}, {len(codes)} cells", flush=True)
n_dim = codes.shape[1]
levels = [int(codes[:, d].max() + 1) for d in range(n_dim)]
print(f"  detected levels per dim: {levels}", flush=True)

# --- [2] Per-dimension marker genes: for each dim d, each level v,
#        log2 fold change vs all other levels, top genes per level
print("\n[2] Per-dimension DE (log2FC vs complement)...", flush=True)
dim_markers = {}
for d in range(n_dim):
    vals = codes[:, d]
    dim_markers[d] = {}
    for v in range(levels[d]):
        mask = vals == v
        if mask.sum() < 30:
            continue
        comp = ~mask
        mu_pos = X[mask].mean(0)
        mu_neg = X[comp].mean(0)
        lfc = np.log2((mu_pos + 1e-6) / (mu_neg + 1e-6))
        top_idx = np.argsort(-lfc)[:20]
        dim_markers[d][v] = {
            "n_cells": int(mask.sum()),
            "top_genes": [hvg_names[i] for i in top_idx],
            "lfc": [float(lfc[i]) for i in top_idx],
            "top_ct": ct_names[int(np.argmax(np.bincount(cell_types[mask], minlength=len(ct_names))))],
            "ct_frac": float(np.bincount(cell_types[mask], minlength=len(ct_names)).max() / mask.sum()),
        }
print("  done", flush=True)

# --- [3] Per-dimension cell-type mapping
print("\n[3] Per-dimension cell-type specificity...", flush=True)
dim_ct = {}
for d in range(n_dim):
    dim_ct[d] = {}
    vals = codes[:, d]
    for v in range(levels[d]):
        mask = vals == v
        if mask.sum() < 30:
            continue
        cts = np.bincount(cell_types[mask], minlength=len(ct_names))
        order = np.argsort(-cts)
        dim_ct[d][v] = [(ct_names[i], int(cts[i]), round(float(cts[i] / mask.sum()), 3))
                        for i in order[:3]]
print("  done", flush=True)

# --- [4] Pathway validation: hypergeometric enrichment of top genes per level
print("\n[4] Pathway / cell-type marker enrichment (hypergeometric)...", flush=True)
markers = {
    "proximal_tubule": ["SLC34A1", "SLC22A6", "SLC5A2", "LRP2", "CUBN", "ALDOB", "HNF4A"],
    "loop_of_Henle_TAL": ["SLC12A1", "UMOD"],
    "DCT": ["SLC12A3"],
    "collecting_duct_principal": ["AQP2", "AQP3", "AVPR2"],
    "intercalated_cell": ["ATP6V0D2", "SLC4A1", "ATP6V1G3"],
    "podocyte": ["NPHS1", "NPHS2", "PODXL", "SYNPO", "WT1"],
    "endothelial": ["PECAM1", "CLDN5", "FLT1", "VWF"],
    "fibroblast": ["COL1A1", "DCN", "PDGFRB"],
    "pericyte": ["RGS5", "NOTCH3"],
    "leukocyte": ["PTPRC", "CD3D", "CD79A"],
}
hvg_set = set(hvg_names)
N_pop = len(hvg_names)

def enrichment(top_genes, gene_set):
    K = len(gene_set)
    top = [g for g in top_genes if g in hvg_set]
    n = len(top)
    x = len([g for g in top if g in gene_set])
    if n == 0 or K == 0:
        return 0, 1.0
    pval = hypergeom.sf(x - 1, N_pop, K, n) if x > 0 else 1.0
    return x, float(pval)

enrich_results = {}
for d in range(n_dim):
    enrich_results[d] = {}
    for v, info in dim_markers[d].items():
        enrich_results[d][v] = {}
        for gs_name, gs in markers.items():
            x, pval = enrichment(info["top_genes"], gs)
            enrich_results[d][v][gs_name] = {"overlap": x, "pval": pval}
print("  done", flush=True)

# --- [5] Interpretable summary ---
print("\n" + "=" * 65, flush=True)
print("INTERPRETABLE SUMMARY", flush=True)
print("=" * 65, flush=True)
for d in range(n_dim):
    print(f"\n### Dimension {d} (levels: {levels[d]})", flush=True)
    for v in range(levels[d]):
        info = dim_markers[d].get(v)
        if info is None:
            continue
        er = enrich_results[d][v]
        best = sorted(er.items(), key=lambda kv: kv[1]["pval"])
        best_str = ", ".join(f"{gs}({r['overlap']}g,p={r['pval']:.1e})"
                             for gs, r in best[:2] if r["overlap"] > 0)
        top5 = ", ".join(info["top_genes"][:5])
        print(f"  Level {v}: n={info['n_cells']}  dominant_ct={info['top_ct']} ({info['ct_frac']:.2f})", flush=True)
        print(f"    top genes: {top5}", flush=True)
        print(f"    enrichment: {best_str if best_str else 'none'}", flush=True)

# --- [6] Save ---
out = {
    "dim_markers": dim_markers,
    "dim_ct": dim_ct,
    "enrichment": enrich_results,
    "n_dim": n_dim,
    "levels": levels,
}
with open(f"{OUTPUT}/interpretability_{RUN}.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}/interpretability_{RUN}.json", flush=True)
