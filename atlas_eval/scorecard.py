"""Disentanglement scorecard for the atlas VQ-VAE.

One command encodes a data bundle under every representation
(:mod:`atlas_eval.representations`), scores it with the shared metric set
(:mod:`atlas_eval.metrics`), and writes a single JSON table covering the atlas
split plus every OOD ladder rung.

    python -m atlas_eval.scorecard \
        --config configs/v6_weak_supervision.yaml \
        --run-id atlas_v6_log1p_s20260809 \
        --target-data-run-ids atlas_train_2608 atlas_ood_unseen_protocol_2608

Read the JSON top down: ``bundles.<bundle>.representations.<view>.summary``
gives the headline ratios, ``verdict`` turns them into pass/fail flags against
documented thresholds, and the nested blocks keep every raw number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

from atlas_eval import metrics as M

if TYPE_CHECKING:  # torch and the vq_2608 tree are only needed to extract, not to score
    from atlas_eval.representations import RepresentationPack

SCHEMA_VERSION = "1.0"

BATCH_LABELS = ("dataset_id", "assay", "donor_id")
BIOLOGY_LABELS = ("coarse_cell_type", "cell_type", "tissue", "disease")
DEFAULT_REPRESENTATIONS = (
    "input_expression",
    "encoder_z_e",
    "bio_z_q",
    "bio_z_q_no_tech",
    "bio_code_onehot",
    "technical_embedding",
    "bio_reconstruction",
    "full_reconstruction",
)
DEFAULT_TARGETS = (
    "atlas_train_2608",
    "atlas_ood_unseen_protocol_2608",
    "atlas_ood_unseen_tissue_2608",
    "atlas_ood_disease_2608",
)
BIO_VIEWS = ("bio_z_q", "bio_code_onehot", "bio_z_q_no_tech")

# Documented defaults for the verdict block; override on the command line.
DEFAULT_THRESHOLDS = {
    "max_relative_dataset_leakage": 0.5,   # bio leakage vs input expression leakage
    "min_relative_biology_retention": 0.9,  # bio cell-type transfer vs input expression
    "min_technical_branch_dataset_leakage": 0.8,
    "max_technical_branch_biology_leakage": 0.2,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Training config, e.g. configs/v6_weak_supervision.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--vq2608-root", default=None)
    parser.add_argument("--atlas-data-run-id", default="atlas_train_2608")
    parser.add_argument("--target-data-run-ids", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--atlas-fit-split", default="train")
    parser.add_argument("--atlas-score-split", default="test")
    parser.add_argument("--ood-splits", nargs="+", default=["train", "val", "test"],
                        help="OOD bundle splits are all unseen by atlas training, so pool them by default.")
    parser.add_argument("--representations", nargs="+", default=list(DEFAULT_REPRESENTATIONS))
    parser.add_argument("--max-cells", type=int, default=100_000)
    parser.add_argument("--max-fit-cells", type=int, default=100_000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--probe-model", default="linear", choices=["linear", "random_forest"])
    parser.add_argument("--probe-min-class-count", type=int, default=32)
    parser.add_argument("--probe-max-train-per-class", type=int, default=1024)
    parser.add_argument("--lisi-cells", type=int, default=20_000)
    parser.add_argument("--lisi-k", type=int, default=90)
    parser.add_argument("--kbet-cells", type=int, default=10_000)
    parser.add_argument("--kbet-k", type=int, default=40)
    parser.add_argument("--skip-neighbor-metrics", action="store_true",
                        help="Skip iLISI/cLISI/kBET (kNN search dominates runtime).")
    parser.add_argument("--transfer-folds", type=int, default=5)
    parser.add_argument("--code-usage-min-cells", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output", default=None, help="Defaults to <run root>/disentanglement_scorecard.json")
    for name in DEFAULT_THRESHOLDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=DEFAULT_THRESHOLDS[name])
    return parser.parse_args(argv)


def _half_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return order[: n // 2], order[n // 2 :]


def score_representation(
    name: str,
    features: np.ndarray,
    pack: "RepresentationPack",
    args: argparse.Namespace,
    fit_pack: Optional["RepresentationPack"],
) -> Dict[str, Any]:
    labels = pack.labels
    fit_indices, score_indices = _half_split(len(features), args.seed)

    def probe(label: str, source: str = "within_bundle") -> Optional[Dict[str, Any]]:
        if label not in labels:
            return None
        if source == "atlas_transfer":
            if fit_pack is None or name not in fit_pack.representations or label not in fit_pack.labels:
                return None
            return M.label_probe(
                fit_pack.representations[name], fit_pack.labels[label], features, labels[label],
                min_class_count=args.probe_min_class_count,
                max_train_per_class=args.probe_max_train_per_class,
                seed=args.seed, model=args.probe_model,
            )
        return M.label_probe(
            features[fit_indices], labels[label][fit_indices],
            features[score_indices], labels[label][score_indices],
            min_class_count=args.probe_min_class_count,
            max_train_per_class=args.probe_max_train_per_class,
            seed=args.seed, model=args.probe_model,
        )

    batch_probes = {label: probe(label) for label in BATCH_LABELS}
    biology_probes = {label: probe(label) for label in BIOLOGY_LABELS}
    transfer_probes = {label: probe(label, "atlas_transfer") for label in BIOLOGY_LABELS}

    neighbours: Dict[str, Any] = {"ilisi": {}, "clisi": {}, "kbet": {}}
    if not args.skip_neighbor_metrics:
        for label in ("dataset_id", "assay"):
            if label in labels:
                neighbours["ilisi"][label] = M.simpson_diversity(
                    features, labels[label], k=args.lisi_k, max_cells=args.lisi_cells, seed=args.seed
                )
                neighbours["kbet"][label] = M.kbet_acceptance(
                    features, labels[label], k=args.kbet_k, max_cells=args.kbet_cells, seed=args.seed
                )
        for label in ("coarse_cell_type", "cell_type"):
            if label in labels:
                neighbours["clisi"][label] = M.simpson_diversity(
                    features, labels[label], k=args.lisi_k, max_cells=args.lisi_cells, seed=args.seed
                )

    cross_dataset = None
    if "coarse_cell_type" in labels and "dataset_id" in labels:
        cross_dataset = M.cross_group_transfer(
            features, labels["coarse_cell_type"], labels["dataset_id"],
            max_folds=args.transfer_folds, min_class_count=args.probe_min_class_count,
            max_train_per_class=args.probe_max_train_per_class, seed=args.seed, model=args.probe_model,
        )

    summary = {
        "dataset_leakage_ratio": M.leakage_ratio(batch_probes.get("dataset_id")),
        "assay_leakage_ratio": M.leakage_ratio(batch_probes.get("assay")),
        "donor_leakage_ratio": M.leakage_ratio(batch_probes.get("donor_id")),
        "coarse_cell_type_readout_ratio": M.leakage_ratio(biology_probes.get("coarse_cell_type")),
        "cross_dataset_cell_type_balanced_accuracy": (
            None if cross_dataset is None else cross_dataset["mean_balanced_accuracy"]
        ),
        "dataset_ilisi_normalized": (
            (neighbours["ilisi"].get("dataset_id") or {}).get("lisi_normalized")
        ),
        "coarse_cell_type_clisi_normalized": (
            (neighbours["clisi"].get("coarse_cell_type") or {}).get("lisi_normalized")
        ),
    }
    return {
        "batch_leakage": {"probes": batch_probes, "ilisi": neighbours["ilisi"], "kbet": neighbours["kbet"]},
        "biology_conservation": {
            "probes": biology_probes,
            "atlas_transfer_probes": transfer_probes,
            "clisi": neighbours["clisi"],
            "cross_dataset_transfer": cross_dataset,
        },
        "summary": summary,
    }


def vq_diagnostics(pack: "RepresentationPack", args: argparse.Namespace) -> Dict[str, Any]:
    labels = pack.labels
    usage = M.codebook_usage(pack.code_indices, pack.meta["codebook_size"])
    conditional = None
    if "dataset_id" in labels and "coarse_cell_type" in labels:
        conditional = M.conditional_code_usage_tv(
            pack.soft_code_usage.astype(np.float32),
            labels["dataset_id"],
            labels["coarse_cell_type"],
            min_cells=args.code_usage_min_cells,
        )
    per_axis: Dict[str, Any] = {}
    for label in ("dataset_id", "assay", "coarse_cell_type", "cell_type"):
        if label in labels:
            per_axis[label] = M.axis_label_association(pack.code_indices, labels[label])

    rows: List[Dict[str, Any]] = []
    tv_by_axis = (conditional or {}).get("per_axis_moved_probability_mass")
    for axis_id in range(pack.meta["num_axes"]):
        row: Dict[str, Any] = {"axis_id": axis_id}
        for label, values in per_axis.items():
            if values is not None:
                row[f"nmi_{label}"] = values[axis_id]["normalized_mutual_info"]
        if tv_by_axis is not None:
            row["conditional_moved_probability_mass"] = float(tv_by_axis[axis_id])
        row["normalized_entropy"] = usage["per_axis"][axis_id]["normalized_entropy"]
        row["dead_codes"] = usage["per_axis"][axis_id]["dead_codes"]
        leak = row.get("nmi_dataset_id")
        bio = row.get("nmi_coarse_cell_type")
        if leak is not None and bio is not None:
            row["batch_minus_biology_nmi"] = float(leak - bio)
        rows.append(row)
    ranked = [row for row in rows if "batch_minus_biology_nmi" in row]
    ranked.sort(key=lambda row: row["batch_minus_biology_nmi"], reverse=True)
    return {
        "codebook_usage": usage,
        "conditional_code_usage_tv": conditional,
        "per_axis": rows,
        "most_batch_leaking_axes": [row["axis_id"] for row in ranked[:5]],
        "most_biological_axes": [row["axis_id"] for row in ranked[-5:]][::-1],
    }


def bundle_verdict(
    representations: Dict[str, Any], args: argparse.Namespace, bio_views: Sequence[str] = BIO_VIEWS
) -> Dict[str, Any]:
    reference = representations.get("input_expression", {}).get("summary", {})
    technical = representations.get("technical_embedding", {}).get("summary", {})
    verdict: Dict[str, Any] = {
        "thresholds": {name: float(getattr(args, name)) for name in DEFAULT_THRESHOLDS},
        "reference_representation": "input_expression",
        "views": {},
    }
    for view in bio_views:
        summary = representations.get(view, {}).get("summary")
        if not summary:
            continue
        checks: Dict[str, Any] = {}
        leak, leak_ref = summary.get("dataset_leakage_ratio"), reference.get("dataset_leakage_ratio")
        if leak is not None and leak_ref:
            checks["relative_dataset_leakage"] = float(leak / leak_ref)
            checks["batch_leakage_reduced"] = bool(leak / leak_ref <= args.max_relative_dataset_leakage)
        bio, bio_ref = (
            summary.get("cross_dataset_cell_type_balanced_accuracy"),
            reference.get("cross_dataset_cell_type_balanced_accuracy"),
        )
        if bio is not None and bio_ref:
            checks["relative_biology_retention"] = float(bio / bio_ref)
            checks["biology_preserved"] = bool(bio / bio_ref >= args.min_relative_biology_retention)
        if checks.get("batch_leakage_reduced") is not None and checks.get("biology_preserved") is not None:
            checks["disentangled"] = bool(checks["batch_leakage_reduced"] and checks["biology_preserved"])
        verdict["views"][view] = checks
    if technical:
        technical_checks: Dict[str, Any] = {}
        if technical.get("dataset_leakage_ratio") is not None:
            technical_checks["encodes_dataset"] = bool(
                technical["dataset_leakage_ratio"] >= args.min_technical_branch_dataset_leakage
            )
        if technical.get("coarse_cell_type_readout_ratio") is not None:
            technical_checks["free_of_biology"] = bool(
                technical["coarse_cell_type_readout_ratio"] <= args.max_technical_branch_biology_leakage
            )
        verdict["technical_branch"] = technical_checks
    return verdict


def score_bundle(
    pack: "RepresentationPack",
    args: argparse.Namespace,
    fit_pack: Optional["RepresentationPack"],
    bio_views: Sequence[str] = BIO_VIEWS,
    include_vq_diagnostics: bool = True,
) -> Dict[str, Any]:
    representations = {}
    for name in args.representations:
        features = pack.representations.get(name)
        if features is None:
            continue
        representations[name] = score_representation(name, features, pack, args, fit_pack)
    bundle: Dict[str, Any] = {
        "splits": pack.meta["splits"],
        "num_cells": pack.meta["num_cells"],
        "num_datasets": int(len(set(M.clean_labels(pack.labels.get("dataset_id", [])).tolist()))),
        "representations": representations,
        "verdict": bundle_verdict(representations, args, bio_views=bio_views),
    }
    if include_vq_diagnostics and pack.code_indices is not None and pack.code_indices.size:
        bundle["vq_diagnostics"] = vq_diagnostics(pack, args)
    return bundle


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    started = time.time()
    from atlas_eval.representations import (
        build_model_for_bundle,
        extract_representations,
        load_atlas_model,
        load_bundle,
    )

    ckpt, config, device, out, checkpoint_meta = load_atlas_model(
        args.config, args.run_id, args.checkpoint, args.vq2608_root
    )
    from clean_common import progress

    atlas_bundle = load_bundle(config, args.atlas_data_run_id)
    model = build_model_for_bundle(ckpt, config, atlas_bundle, device)

    progress(f"extracting atlas fitting representations from {args.atlas_data_run_id}/{args.atlas_fit_split}")
    fit_pack = extract_representations(
        model, config, atlas_bundle, device,
        splits=[args.atlas_fit_split], max_cells=args.max_fit_cells,
        projection_dim=args.projection_dim, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "checkpoint": checkpoint_meta,
        "atlas_data_run_id": args.atlas_data_run_id,
        "settings": {
            key: value for key, value in vars(args).items() if key not in {"config", "run_id"}
        },
        "config_path": args.config,
        "bundles": {},
    }

    for data_run_id in args.target_data_run_ids:
        is_atlas = data_run_id == args.atlas_data_run_id
        splits = [args.atlas_score_split] if is_atlas else args.ood_splits
        progress(f"scoring bundle {data_run_id} splits={splits}")
        bundle = atlas_bundle if is_atlas else load_bundle(config, data_run_id)
        pack = extract_representations(
            model, config, bundle, device,
            splits=splits, max_cells=args.max_cells,
            projection_dim=args.projection_dim, seed=args.seed,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        # The atlas fit pack always comes from the training split, so using it as
        # the readout for the held-out atlas test cells is transfer, not leakage.
        payload["bundles"][data_run_id] = score_bundle(pack, args, fit_pack)
        del pack

    payload["runtime_seconds"] = float(time.time() - started)
    output = Path(args.output) if args.output else out["root"] / "disentanglement_scorecard.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    progress(f"wrote scorecard to {output}")
    return payload


if __name__ == "__main__":
    main()
