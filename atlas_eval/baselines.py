"""Train scVI / scANVI baselines on atlas bundles and score them like the VQ-VAE.

The point of the baseline runner is comparability, not novelty: it reads the
exact same records (same cells, same gene vocabulary, same ``H5CSRDataset``
path as the project's own model), trains scVI (or scANVI) on raw counts with
``dataset_id`` as the batch key, encodes every target bundle into a single
latent view, and feeds that latent through the shared metric set
(:mod:`atlas_eval.scorecard`). The output JSON has the same bundle schema so
the same report tooling applies; the verdict is computed on the ``latent``
view against the ``input_expression`` reference.

    python -m atlas_eval.baselines \
        --config configs/v6_weak_supervision.yaml \
        --method scvi \
        --fit-data-run-id atlas_bt5_train \
        --target-data-run-ids atlas_bt5_matched_ood atlas_ood_unseen_protocol_2608 \
        --run-id scvi_bt5 \
        --max-cells 40000 --max-fit-cells 40000 --train-max-cells 250000

Notes on fairness:

- the latent is the scVI posterior mean (deterministic at eval time);
- ``dataset_id`` is the batch key, exactly the technical factor our model is
  asked to strip from its biological codes;
- scANVI additionally consumes ``coarse_cell_type`` labels (its semi-supervised
  variant), which the VQ-VAE also sees at training time;
- early stopping runs on the held-out validation split when available.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from atlas_eval.paths import ensure_on_path
from atlas_eval.representations import LABEL_FIELDS, RepresentationPack

BIO_VIEW = "latent"
SCHEMA_VERSION = "1.0"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Training config; resolves output.root.")
    parser.add_argument("--method", required=True, choices=["scvi", "scanvi"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fit-data-run-id", required=True, help="Bundle used to train the baseline.")
    parser.add_argument("--target-data-run-ids", nargs="+", default=["atlas_train_2608"])
    parser.add_argument("--fit-splits", nargs="+", default=["train"])
    parser.add_argument("--val-split", default="val", help="Validation split for early stopping ('' disables).")
    parser.add_argument("--score-splits", nargs="+", default=["train", "val", "test"],
                        help="Target bundle splits to pool for scoring.")
    parser.add_argument("--representations", nargs="+", default=["input_expression", "latent"],
                        help="Views to score (must include the reference and the latent).")
    parser.add_argument("--max-cells", type=int, default=100_000)
    parser.add_argument("--max-fit-cells", type=int, default=100_000)
    parser.add_argument("--train-max-cells", type=int, default=250_000,
                        help="Cap on training cells (keeps baseline wall-clock sane at atlas scale).")
    parser.add_argument("--n-latent", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--probe-model", default="linear", choices=["linear", "random_forest"])
    parser.add_argument("--probe-min-class-count", type=int, default=32)
    parser.add_argument("--probe-max-train-per-class", type=int, default=1024)
    parser.add_argument("--lisi-cells", type=int, default=20_000)
    parser.add_argument("--lisi-k", type=int, default=90)
    parser.add_argument("--kbet-cells", type=int, default=10_000)
    parser.add_argument("--kbet-k", type=int, default=40)
    parser.add_argument("--skip-neighbor-metrics", action="store_true")
    parser.add_argument("--transfer-folds", type=int, default=5)
    parser.add_argument("--output", default=None, help="Defaults to <run root>/<run-id>/baseline_scorecard.json")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip training; load saved model from <run root>/<run-id>/<method>_model/")
    from atlas_eval.scorecard import DEFAULT_THRESHOLDS

    for name in DEFAULT_THRESHOLDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=DEFAULT_THRESHOLDS[name])
    return parser.parse_args(argv)


def load_config_and_root(config_path: str, run_id: str):
    ensure_on_path()
    from vq_pipeline.runtime import load_config, make_output_dirs

    config = load_config(config_path)
    out = make_output_dirs(config, run_id)
    return config, out


def _counts_config(config: Dict[str, Any]) -> Dict[str, Any]:
    local = dict(config)
    local["data"] = dict(config.get("data", {}))
    local["data"]["return_raw_counts"] = True
    return local


def build_anndata(
    config: Dict[str, Any],
    data_run_id: str,
    splits: Sequence[str],
    max_cells: int,
    seed: int,
    chunk: int = 4096,
) -> Any:
    """Read raw counts + labels into an AnnData for scVI.

    Reuses ``H5CSRDataset`` (through the same DataLoader path as the model
    training loop) so row selection and the gene map are identical to the
    model training path; only the count matrix is read (``return_raw_counts``),
    and the deterministic max-cell cap mirrors ``extract_representations``.
    """
    import scipy.sparse as sp

    from atlas_eval.representations import load_bundle

    ensure_on_path()
    from torch.utils.data import ConcatDataset, DataLoader

    from vq_pipeline.data import H5CSRDataset

    bundle = load_bundle(config, data_run_id)
    split_names = [name for name in splits if name in bundle["splits"]]
    if not split_names:
        raise ValueError(f"bundle {data_run_id} has none of the splits {list(splits)}")
    datasets = [
        H5CSRDataset(bundle["splits"][name], bundle["file_gene_maps"], bundle["gene_vocab"],
                     _counts_config(config), bundle["file_metas"])
        for name in split_names
    ]
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    total = len(combined)
    if max_cells and total > int(max_cells):
        rng = np.random.default_rng(seed)
        from torch.utils.data import Subset

        chosen = np.sort(rng.choice(total, size=int(max_cells), replace=False)).tolist()
        combined = Subset(combined, chosen)
    loader = DataLoader(combined, batch_size=chunk, shuffle=False, num_workers=0)

    rows: List[Any] = []
    obs: Dict[str, List[Any]] = {name: [] for name in LABEL_FIELDS}
    started = time.time()
    seen = 0
    for item in loader:
        counts = item["x_counts"].numpy().astype(np.float32)
        for row in counts:
            rows.append(sp.csr_matrix(row))
        for name in obs:
            values = item.get(name)
            if values is None:
                obs[name].extend(["unknown"] * int(counts.shape[0]))
            else:
                obs[name].extend(list(values))
        seen += int(counts.shape[0])
        if seen % 10_000 == 0:
            print(f"[build_anndata {data_run_id}] {seen} cells ({time.time() - started:.0f}s)", flush=True)
    for dataset in datasets:
        dataset.close()

    import anndata as ad

    matrix = sp.vstack(rows, format="csr")
    adata = ad.AnnData(X=matrix)
    adata.var_names = list(bundle["gene_vocab"])
    for name, values in obs.items():
        adata.obs[name] = np.asarray(values, dtype=object)
    adata.uns["_bundle"] = data_run_id
    return adata


def train_baseline(
    adata: Any,
    method: str,
    *,
    val_adata: Optional[Any] = None,
    n_latent: int = 256,
    max_epochs: int = 120,
    patience: int = 10,
    batch_size: Optional[int] = None,
    seed: int = 1701,
) -> Tuple[Any, Dict[str, Any]]:
    import scvi

    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata, batch_key="dataset_id")
    scvi_model = scvi.model.SCVI(adata, n_latent=int(n_latent))
    early = bool(val_adata is not None and len(val_adata) > 0)
    scvi_model.train(
        max_epochs=int(max_epochs),
        early_stopping=early,
        early_stopping_patience=int(patience),
        early_stopping_monitor="validation_loss",
        plan_kwargs={},
        batch_size=batch_size or 128,
    )
    history = scvi_model.history
    n_epochs = int(len(history))
    info: Dict[str, Any] = {
        "method": method,
        "n_latent": int(n_latent),
        "epochs_run": n_epochs,
        "train_cells": int(adata.n_obs),
        "early_stopping": early,
        "best_val_loss": float(history["validation_loss"].min()) if early and "validation_loss" in history else None,
        "final_train_loss": float(history["train_loss_epoch"].iloc[-1]) if "train_loss_epoch" in history else None,
    }
    if method == "scanvi":
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model, unlabeled_category="unknown", labels_key="coarse_cell_type"
        )
        scanvi_model.train(max_epochs=int(max_epochs), early_stopping=early,
                           early_stopping_patience=int(patience), batch_size=batch_size or 128)
        info["scanvi_epochs_run"] = int(len(scanvi_model.history))
        return scanvi_model, info
    return scvi_model, info


def load_baseline(method: str, model_dir: Path, adata: Optional[Any] = None) -> Any:
    import scvi

    if method == "scanvi":
        model = scvi.model.SCANVI.load(str(model_dir), adata=adata)
    else:
        model = scvi.model.SCVI.load(str(model_dir), adata=adata)
    return model


def _project(features: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim <= 0 or features.shape[1] <= dim:
        return features
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((features.shape[1], dim)) / np.sqrt(dim)
    return features @ matrix


def _remap_unseen_labels(adata: Any, model: Any) -> None:
    """Rewrite scANVI label values unseen at training time to the unlabeled category.

    scANVI's label field hard-codes ``extend_categories=False`` on transfer
    ("don't extend labels for query data"), so query bundles containing cell
    types absent from the training registry would crash inference.  Echoing the
    intended semantics of ``unlabeled_category``, remap those to ``unknown``.
    """
    import pandas as pd

    registry = model.adata_manager.registry
    label_state = registry["field_registries"]["labels"]["state_registry"]
    known = np.asarray(list(label_state["categorical_mapping"]))
    key = label_state["original_key"]
    if key in adata.obs:
        col = np.asarray(adata.obs[key].values, dtype=object)
        unseen = ~pd.Index(col).isin(known)
        if unseen.any():
            col[unseen] = str(label_state["unlabeled_category"])
            adata.obs[key] = col
            print(f"[encode_bundle] remapped {int(unseen.sum())} unseen labels to 'unknown'", flush=True)


def encode_bundle(
    model: Any,
    config: Dict[str, Any],
    data_run_id: str,
    splits: Sequence[str],
    max_cells: int,
    seed: int,
    projection_dim: int,
) -> RepresentationPack:
    """Encode one bundle: scVI latent plus a deterministic log1p reference view."""
    adata = build_anndata(config, data_run_id, splits, max_cells, seed)
    try:
        if model.__class__.__name__ == "SCANVI":
            _remap_unseen_labels(adata, model)
        manager = model.adata_manager.transfer_fields(adata, extend_categories=True)
        model._register_manager_for_instance(manager)
    except Exception:  # noqa: BLE001
        pass
    latent = np.asarray(model.get_latent_representation(adata)).astype(np.float32)
    labels = {name: np.asarray(adata.obs[name].values, dtype=object) for name in LABEL_FIELDS
              if name in adata.obs}

    counts = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    counts = np.asarray(counts, dtype=np.float32)
    sums = np.asarray(counts.sum(axis=1)).ravel()
    normalized = counts / (sums[:, None] / 10_000.0)
    normalized = np.log1p(normalized)
    if hasattr(normalized, "toarray"):
        normalized = normalized.toarray()
    normalized = np.asarray(normalized, dtype=np.float32)

    pack = RepresentationPack(
        representations={
            "input_expression": _project(normalized, projection_dim, seed),
            BIO_VIEW: _project(latent, projection_dim, seed),
        },
        labels=labels,
        code_indices=np.empty((int(latent.shape[0]), 0), dtype=np.int64),
        soft_code_usage=np.empty((int(latent.shape[0]), 0), dtype=np.float32),
        meta={
            "splits": list(splits),
            "num_cells": int(latent.shape[0]),
            "projection_dim": int(projection_dim),
        },
    )
    return pack


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    started = time.time()
    config, out = load_config_and_root(args.config, args.run_id)
    out_root = Path(out["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[baselines {args.method}] building train anndata from {args.fit_data_run_id}/{args.fit_splits}")
    train_adata = build_anndata(
        config, args.fit_data_run_id, args.fit_splits, args.train_max_cells, args.seed
    )
    val_adata = None
    if args.val_split:
        try:
            val_adata = build_anndata(config, args.fit_data_run_id, [args.val_split], args.max_fit_cells, args.seed)
            print(f"[baselines] validation anndata: {val_adata.n_obs} cells")
        except Exception as error:  # noqa: BLE001
            print(f"[baselines] no validation split ({error}); training without early stopping")

    model_dir = out_root / f"{args.method}_model"
    if args.score_only:
        print(f"[baselines {args.method}] score-only: loading model from {model_dir}")
        model = load_baseline(args.method, model_dir, adata=train_adata)
        info: Dict[str, Any] = {"method": args.method, "n_latent": int(args.n_latent),
                                "epochs_run": None, "train_cells": None,
                                "early_stopping": None, "score_only": True}
    else:
        model, info = train_baseline(
            train_adata, args.method, val_adata=val_adata,
            n_latent=args.n_latent, max_epochs=args.max_epochs,
            patience=args.early_stopping_patience, batch_size=args.batch_size, seed=args.seed,
        )
        print(f"[baselines] training done: {info}")
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir), overwrite=True, save_anndata=True)
        print(f"[baselines] saved model to {model_dir}")

    from atlas_eval import scorecard as SC

    fit_pack = encode_bundle(model, config, args.fit_data_run_id, args.fit_splits,
                             args.max_fit_cells, args.seed, args.projection_dim)
    print(f"[baselines] fit pack: {fit_pack.meta['num_cells']} cells")

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "model_type": args.method,
        "atlas_data_run_id": args.fit_data_run_id,
        "training": info,
        "settings": {key: value for key, value in vars(args).items() if key not in {"config", "run_id"}},
        "config_path": args.config,
        "bundles": {},
    }

    for data_run_id in args.target_data_run_ids:
        pack = encode_bundle(model, config, data_run_id, args.score_splits,
                             args.max_cells, args.seed, args.projection_dim)
        bundle = SC.score_bundle(
            pack, args, fit_pack, bio_views=(BIO_VIEW,), include_vq_diagnostics=False
        )
        payload["bundles"][data_run_id] = bundle
        print(f"[baselines] scored {data_run_id}: {pack.meta['num_cells']} cells", flush=True)
        del pack

    payload["runtime_seconds"] = float(time.time() - started)
    output = Path(args.output) if args.output else out_root / "baseline_scorecard.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"[baselines] wrote {output}")
    return payload


if __name__ == "__main__":
    main()