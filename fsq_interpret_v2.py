import os, sys, json
import numpy as np
from scipy.stats import hypergeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT = "/home/aly/vqvae-batch/output/fsq"
RUN = sys.argv[1] if len(sys.argv) > 1 else "L3x8_ct1.0"

print("=" * 65, flush=True)
print(f"Interpretability + pathway validation for FSQ-VAE v2 [{RUN}]", flush=True)
print("=" * 65, flush=True)

d = np.load(f"{OUTPUT}/cached_data.npz")
X, batches, cell_types = d["X"], d["batches"], d["cell_types"]
hvg_names = np.array(d["gene_names"])
n_ct = int(d["n_cell_types"])
ct_names = [str(i) for i in range(n_ct)]
print(f"  {X.shape[0]} cells, {len(hvg_names)} genes", flush=True)

codes = np.load(f"{OUTPUT}/codes_{RUN}.npy")
code_id = np.load(f"{OUTPUT}/code_id_{RUN}.npy").ravel()
n_dim = codes.shape[1]
levels = [int(codes[:, d].max() + 1) for d in range(n_dim)]
print(f"  codes: {codes.shape}, levels per dim: {levels}", flush=True)

print("\n[1] Per-dimension DE (log2FC vs complement)...", flush=True)
dim_markers = {}
for dd in range(n_dim):
    vals = codes[:, dd]
    dim_markers[dd] = {}
    for v in range(levels[dd]):
        mask = vals == v
        if mask.sum() < 30:
            continue
        comp = ~mask
        mu_pos = X[mask].mean(0)
        mu_neg = X[comp].mean(0)
        lfc = np.log2((mu_pos + 1e-6) / (mu_neg + 1e-6))
        top_idx = np.argsort(-lfc)[:20]
        dim_markers[dd][v] = {
            "n_cells": int(mask.sum()),
            "top_genes": [hvg_names[i] for i in top_idx],
            "lfc": [float(lfc[i]) for i in top_idx],
            "top_ct": ct_names[int(np.argmax(np.bincount(cell_types[mask], minlength=n_ct)))],
            "ct_frac": float(np.bincount(cell_types[mask], minlength=n_ct).max() / mask.sum()),
        }
print("  done", flush=True)

print("\n[2] Cell-type specificity per level...", flush=True)
dim_ct = {}
for dd in range(n_dim):
    dim_ct[dd] = {}
    vals = codes[:, dd]
    for v in range(levels[dd]):
        mask = vals == v
        if mask.sum() < 30:
            continue
        cts = np.bincount(cell_types[mask], minlength=n_ct)
        order = np.argsort(-cts)
        dim_ct[dd][v] = [(ct_names[i], int(cts[i]), round(float(cts[i] / mask.sum()), 3))
                         for i in order[:3]]
print("  done", flush=True)

print("\n[3] Pathway enrichment (hypergeometric vs kidney markers)...", flush=True)
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
for dd in range(n_dim):
    enrich_results[dd] = {}
    for v, info in dim_markers[dd].items():
        enrich_results[dd][v] = {}
        for gs_name, gs in markers.items():
            x, pval = enrichment(info["top_genes"], gs)
            enrich_results[dd][v][gs_name] = {"overlap": x, "pval": pval}
print("  done", flush=True)

print("\n" + "=" * 65, flush=True)
print("INTERPRETABLE SUMMARY", flush=True)
print("=" * 65, flush=True)
for dd in range(n_dim):
    print(f"\n### Dimension {dd} (levels: {levels[dd]})", flush=True)
    for v in range(levels[dd]):
        info = dim_markers[dd].get(v)
        if info is None:
            continue
        er = enrich_results[dd][v]
        best = sorted(er.items(), key=lambda kv: kv[1]["pval"])
        best_str = ", ".join(f"{gs}({r['overlap']}g,p={r['pval']:.1e})"
                             for gs, r in best[:2] if r["overlap"] > 0)
        top5 = ", ".join(info["top_genes"][:5])
        print(f"  Level {v}: n={info['n_cells']}  dominant_ct={info['top_ct']} ({info['ct_frac']:.2f})", flush=True)
        print(f"    top genes: {top5}", flush=True)
        print(f"    enrichment: {best_str if best_str else 'none'}", flush=True)

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
