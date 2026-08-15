"""Build a biology-matched subset for disentanglement stress tests.

Given one h5ad, this script filters cells by specified biological constraints
then picks top batches/datasets with enough cells. It writes:
1) a filtered h5ad subset,
2) a JSON summary manifest for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def _parse_csv(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_constraints(raw: str) -> Dict[str, List[str]]:
    """Parse 'k=v1|v2,k2=v3' style strings."""
    out: Dict[str, List[str]] = {}
    if not raw:
        return out
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid constraint '{chunk}'. Expected key=value1|value2.")
        k, v = chunk.split("=", 1)
        vals = [item.strip() for item in v.split("|") if item.strip()]
        if not vals:
            raise ValueError(f"Constraint '{chunk}' has no values.")
        out[k.strip()] = vals
    return out


def _value_counts(obs_col) -> Dict[str, int]:
    vc = obs_col.astype(str).value_counts()
    return {str(k): int(v) for k, v in vc.items()}


def main():
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Input .h5ad")
    ap.add_argument("--output-h5ad", required=True, help="Filtered output .h5ad")
    ap.add_argument("--output-manifest", default="", help="Optional JSON manifest path")
    ap.add_argument("--batch-field", default="dataset_id")
    ap.add_argument("--celltype-field", default="cell_type")
    ap.add_argument(
        "--constraints",
        default="",
        help="Biology constraints: key=v1|v2,key2=v3. Example: tissue=lung,disease=healthy",
    )
    ap.add_argument("--min-cells-per-batch", type=int, default=1000)
    ap.add_argument("--max-batches", type=int, default=30)
    ap.add_argument("--max-cells", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    constraints = _parse_constraints(args.constraints)

    adata = ad.read_h5ad(args.data_path)
    initial_cells = int(adata.n_obs)
    if args.batch_field not in adata.obs.columns:
        raise ValueError(f"batch_field '{args.batch_field}' not found.")
    if args.celltype_field not in adata.obs.columns:
        raise ValueError(f"celltype_field '{args.celltype_field}' not found.")

    mask = np.ones(adata.n_obs, dtype=bool)
    for field, values in constraints.items():
        if field not in adata.obs.columns:
            raise ValueError(f"constraint field '{field}' not found.")
        m = adata.obs[field].astype(str).isin(values).to_numpy()
        mask &= m

    adata = adata[mask].copy()

    # Keep batches with enough cells; then keep top N by size.
    bc = adata.obs[args.batch_field].astype(str).value_counts()
    keep = bc[bc >= args.min_cells_per_batch].index.tolist()
    if len(keep) > args.max_batches:
        keep = bc.loc[keep].nlargest(args.max_batches).index.tolist()
    adata = adata[adata.obs[args.batch_field].astype(str).isin(keep)].copy()

    if adata.n_obs == 0:
        raise ValueError("No cells remain after constraints and batch filtering.")

    if args.max_cells and adata.n_obs > args.max_cells:
        rng = np.random.RandomState(args.seed)
        pick = rng.choice(adata.n_obs, args.max_cells, replace=False)
        adata = adata[pick].copy()

    out_h5ad_dir = os.path.dirname(args.output_h5ad)
    if out_h5ad_dir:
        os.makedirs(out_h5ad_dir, exist_ok=True)
    adata.write_h5ad(args.output_h5ad)

    manifest = {
        "input_path": os.path.abspath(args.data_path),
        "output_path": os.path.abspath(args.output_h5ad),
        "initial_cells": initial_cells,
        "final_cells": int(adata.n_obs),
        "final_genes": int(adata.n_vars),
        "batch_field": args.batch_field,
        "celltype_field": args.celltype_field,
        "constraints": constraints,
        "min_cells_per_batch": args.min_cells_per_batch,
        "max_batches": args.max_batches,
        "max_cells": args.max_cells,
        "batch_counts": _value_counts(adata.obs[args.batch_field]),
        "celltype_counts": _value_counts(adata.obs[args.celltype_field]),
    }

    manifest_path = args.output_manifest
    if not manifest_path:
        base, _ = os.path.splitext(args.output_h5ad)
        manifest_path = f"{base}.manifest.json"
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
