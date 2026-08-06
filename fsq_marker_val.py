import os, sys, json
import numpy as np
from scipy.stats import hypergeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUTPUT = "/home/aly/vqvae-batch/output/fsq"
RUN = sys.argv[1] if len(sys.argv) > 1 else "L3x8_ct5.0"

print("=" * 65, flush=True)
print(f"FOCUSED marker validation for FSQ-VAE v2 [{RUN}]", flush=True)
print("=" * 65, flush=True)

d = np.load(f"{OUTPUT}/cached_data.npz")
X, batches, cell_types = d["X"], d["batches"], d["cell_types"]
hvg_names = np.array(d["gene_names"])
n_ct = int(d["n_cell_types"])

codes = np.load(f"{OUTPUT}/codes_{RUN}.npy")
code_id = np.load(f"{OUTPUT}/code_id_{RUN}.npy").ravel()

markers = {
    "endothelial": ["PECAM1", "CLDN5", "FLT1", "VWF"],
    "proximal_tubule": ["SLC34A1", "SLC22A6", "SLC5A2", "LRP2", "CUBN", "ALDOB", "HNF4A"],
    "intercalated_cell": ["ATP6V0D2", "SLC4A1", "ATP6V1G3"],
    "collecting_duct_principal": ["AQP2", "AQP3", "AVPR2"],
    "loop_of_Henle_TAL": ["SLC12A1", "UMOD"],
    "DCT": ["SLC12A3"],
    "fibroblast": ["COL1A1", "DCN", "PDGFRB"],
    "podocyte": ["NPHS1", "NPHS2", "PODXL", "SYNPO", "WT1"],
    "leukocyte": ["PTPRC", "CD3D", "CD79A"],
    "pericyte": ["RGS5", "NOTCH3"],
}
gene_idx = {g: i for i, g in enumerate(hvg_names)}
missing = [g for gs in markers.values() for g in gs if g not in gene_idx]
print(f"markers missing from HVG set: {missing}", flush=True)

# For each code, dominant cell type and mean expression of that type's markers
print("\n[1] Per-code marker upregulation (dominant cell type markers vs rest)...", flush=True)
results = {}
marker_score_rows = []
for c in np.unique(code_id):
    mask = (code_id == c)
    if mask.sum() < 30:
        continue
    cts = np.bincount(cell_types[mask], minlength=n_ct)
    dom = int(np.argmax(cts))
    dom_frac = float(cts.max() / mask.sum())
    gs_name = None
    for name, gs in markers.items():
        if dom in [int(i) for i in range(n_ct)]:
            pass
    # map dominant ct index to marker set (ct names are the annotation labels 0-15)
    # use majority cell type index -> marker list keyed by type order
    marker_list = None
    mkey = None
    # match via ordering: use the loader's celltype_classes
    ct_order = ["endothelial cell", "epithelial cell of proximal tubule",
                "kidney collecting duct intercalated cell", "kidney collecting duct principal cell",
                "kidney connecting tubule epithelial cell",
                "kidney distal convoluted tubule epithelial cell",
                "kidney interstitial fibroblast",
                "kidney loop of Henle thick ascending limb epithelial cell",
                "kidney loop of Henle thin ascending limb epithelial cell",
                "kidney loop of Henle thin descending limb epithelial cell",
                "leukocyte", "neural cell", "papillary tips cell",
                "parietal epithelial cell", "podocyte", "renal interstitial pericyte"]
    name = ct_order[dom] if dom < len(ct_order) else f"ct{dom}"
    # best-effort marker set for the dominant type
    marker_list = None
    for mname, gs in markers.items():
        if mname in name:
            marker_list = gs
            mkey = mname
            break
    if marker_list is None:
        # nearest: use known aliases
        aliases = {
            "epithelial cell of proximal tubule": "proximal_tubule",
            "kidney collecting duct intercalated cell": "intercalated_cell",
            "kidney collecting duct principal cell": "collecting_duct_principal",
            "kidney loop of Henle thick ascending limb epithelial cell": "loop_of_Henle_TAL",
            "kidney distal convoluted tubule epithelial cell": "DCT",
            "kidney interstitial fibroblast": "fibroblast",
            "kidney loop of Henle thin ascending limb epithelial cell": "loop_of_Henle_TAL",
            "endothelial cell": "endothelial",
            "leukocyte": "leukocyte",
            "renal interstitial pericyte": "pericyte",
            "podocyte": "podocyte",
        }
        mkey = aliases.get(name)
        marker_list = markers.get(mkey)
    if marker_list is None:
        continue
    gi = [gene_idx[g] for g in marker_list if g in gene_idx]
    if not gi:
        continue
    comp = ~mask
    mu_pos = X[mask][:, gi].mean()
    mu_neg = X[comp][:, gi].mean()
    lfc = float(np.log2((mu_pos + 1e-6) / (mu_neg + 1e-6)))
    # hypergeometric: are markers enriched in the code's top-50 genes?
    top50 = np.argsort(-(X[mask].mean(0) - X[comp].mean(0)))[:50]
    top_names = set(hvg_names[top50])
    overlap = len(top_names & set(marker_list))
    pval = float(hypergeom.sf(overlap - 1, len(hvg_names), len(marker_list), 50)) if overlap > 0 else 1.0
    results[int(c)] = {
        "n": int(mask.sum()), "dominant_ct": name, "dom_frac": dom_frac,
        "marker_set": mkey, "marker_lfc": lfc,
        "marker_overlap_top50": overlap, "marker_pval": pval,
    }
    marker_score_rows.append((dom_frac, lfc, overlap, pval, name))

# Summary stats
lfc_vals = [r["marker_lfc"] for r in results.values() if r["marker_lfc"] == r["marker_lfc"]]
up = [r for r in results.values() if r["marker_lfc"] > 0.5]
sig = [r for r in results.values() if r["marker_pval"] < 0.05 and r["marker_overlap_top50"] > 0]
print(f"  Total codes with markers: {len(results)}", flush=True)
print(f"  Codes with marker LFC > 0.5: {len(up)} / {len(results)}", flush=True)
print(f"  Codes with significant marker enrichment in top-50: {len(sig)} / {len(results)}", flush=True)
print(f"  Mean marker LFC: {np.mean(lfc_vals):.3f}", flush=True)

print("\n[2] Per-code detail (sorted by marker LFC):", flush=True)
for c, r in sorted(results.items(), key=lambda kv: -kv[1]["marker_lfc"]):
    print(f"  code={c} ct={r['dominant_ct']}({r['dom_frac']:.2f}) "
          f"marker={r['marker_set']} LFC={r['marker_lfc']:+.2f} "
          f"top50_overlap={r['marker_overlap_top50']} p={r['marker_pval']:.1e}", flush=True)

with open(f"{OUTPUT}/marker_validation_{RUN}.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUTPUT}/marker_validation_{RUN}.json", flush=True)
