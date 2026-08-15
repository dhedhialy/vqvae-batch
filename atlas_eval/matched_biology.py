"""Build a biology-matched training subset and its matched OOD holdout.

The controlled experiment behind this module: if the datasets a model trains on
share tissue, disease status, age band and sex mix, then most of what still
separates them is technical.  A model trained there, evaluated on *unseen but
equally matched* datasets, tells us whether the learned batch factors transfer
or were dataset-specific quirks.

The builder reads the manifest CSVs of an existing data bundle, summarises each
dataset's biology, applies explicit inclusion criteria, splits the survivors
into a training subset and a matched-OOD holdout, and (optionally) materialises
both as new bundles that ``src/train.py --data-run-id`` can consume directly.

    python -m atlas_eval.matched_biology \
        --config configs/v6_weak_supervision.yaml \
        --source-data-run-id atlas_train_2608 \
        --name atlas_matched_biology_v1 --write-bundles
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

HEALTHY_VALUES = {"normal", "healthy", "control", "none"}
MATCHING_FIELDS = ("tissue", "disease", "sex", "assay", "coarse_cell_type", "suspension_type")


@dataclass
class DatasetSummary:
    dataset_id: str
    num_cells: int = 0
    file_paths: set = field(default_factory=set)
    donors: set = field(default_factory=set)
    ages: List[float] = field(default_factory=list)
    counters: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def fraction(self, field_name: str, values: Iterable[str]) -> float:
        counter = self.counters.get(field_name)
        if not counter:
            return 0.0
        wanted = {value.lower() for value in values}
        hits = sum(count for label, count in counter.items() if label.lower() in wanted)
        return hits / max(sum(counter.values()), 1)

    def dominant(self, field_name: str) -> Tuple[str, float]:
        counter = self.counters.get(field_name)
        if not counter:
            return "unknown", 0.0
        label, count = counter.most_common(1)[0]
        return label, count / max(sum(counter.values()), 1)

    def median_age(self) -> Optional[float]:
        if not self.ages:
            return None
        ordered = sorted(self.ages)
        return float(ordered[len(ordered) // 2])

    def to_json(self) -> Dict[str, Any]:
        tissue, tissue_fraction = self.dominant("tissue")
        assay, assay_fraction = self.dominant("assay")
        sex, sex_fraction = self.dominant("sex")
        return {
            "dataset_id": self.dataset_id,
            "num_cells": self.num_cells,
            "num_donors": len(self.donors),
            "num_files": len(self.file_paths),
            "dominant_tissue": tissue,
            "dominant_tissue_fraction": tissue_fraction,
            "dominant_assay": assay,
            "dominant_assay_fraction": assay_fraction,
            "dominant_sex": sex,
            "dominant_sex_fraction": sex_fraction,
            "healthy_fraction": self.fraction("disease", HEALTHY_VALUES),
            "median_age_years": self.median_age(),
            "num_coarse_cell_types": len(self.counters.get("coarse_cell_type", {})),
            "assays": dict(self.counters.get("assay", {})),
            "tissues": dict(self.counters.get("tissue", {})),
        }


def bundle_path(output_root: Path, data_run_id: str) -> Path:
    root = Path(output_root) / data_run_id
    slim = root / "data_bundle.slim.json"
    return slim if slim.exists() else root / "data_bundle.json"


def read_bundle_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_relative(path: Path, value: str) -> str:
    candidate = Path(value)
    return str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())


def summarize_datasets(split_paths: Dict[str, str]) -> Dict[str, DatasetSummary]:
    summaries: Dict[str, DatasetSummary] = {}
    for split, path in split_paths.items():
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                dataset_id = row.get("dataset_id") or "unknown"
                summary = summaries.setdefault(dataset_id, DatasetSummary(dataset_id))
                summary.num_cells += 1
                summary.file_paths.add(row["file_path"])
                donor = (row.get("donor_id") or "").strip()
                if donor and donor.lower() not in {"unknown", "na"}:
                    summary.donors.add(donor)
                age = (row.get("age") or "").strip()
                if age:
                    try:
                        summary.ages.append(float(age))
                    except ValueError:
                        pass
                for name in MATCHING_FIELDS:
                    value = (row.get(name) or "unknown").strip() or "unknown"
                    summary.counters[name][value] += 1
    return summaries


def choose_tissues(summaries: Dict[str, DatasetSummary], args: argparse.Namespace) -> List[str]:
    if args.tissue:
        return [value.lower() for value in args.tissue]
    counts = Counter()
    for summary in summaries.values():
        if summary.num_cells < args.min_cells_per_dataset:
            continue
        tissue, fraction = summary.dominant("tissue")
        if tissue != "unknown" and fraction >= args.min_dominant_fraction:
            counts[tissue.lower()] += 1
    if not counts:
        return []
    best = counts.most_common(args.num_tissues)
    return [tissue for tissue, _ in best]


def select_datasets(
    summaries: Dict[str, DatasetSummary], args: argparse.Namespace
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    tissues = choose_tissues(summaries, args)
    rejected: List[Dict[str, Any]] = []
    eligible: List[DatasetSummary] = []
    for summary in summaries.values():
        tissue, tissue_fraction = summary.dominant("tissue")
        reasons = []
        if summary.num_cells < args.min_cells_per_dataset:
            reasons.append("too_few_cells")
        if tissues and tissue.lower() not in tissues:
            reasons.append("tissue_not_matched")
        if tissue_fraction < args.min_dominant_fraction:
            reasons.append("tissue_not_homogeneous")
        if summary.fraction("disease", HEALTHY_VALUES) < args.min_healthy_fraction:
            reasons.append("not_healthy_enough")
        median_age = summary.median_age()
        if args.require_age and median_age is None:
            reasons.append("age_unknown")
        if median_age is not None and not (args.age_min <= median_age <= args.age_max):
            reasons.append("age_out_of_band")
        if len(summary.counters.get("coarse_cell_type", {})) < args.min_coarse_cell_types:
            reasons.append("too_few_cell_types")
        if reasons:
            rejected.append({"dataset_id": summary.dataset_id, "reasons": reasons, **summary.to_json()})
        else:
            eligible.append(summary)

    eligible.sort(key=lambda summary: summary.num_cells, reverse=True)
    if args.max_datasets and len(eligible) > args.max_datasets:
        eligible = eligible[: args.max_datasets]

    # Hold out whole datasets, balanced across assays, so the holdout probes
    # pure batch transfer rather than an unseen protocol by accident.
    by_assay: Dict[str, List[DatasetSummary]] = defaultdict(list)
    for summary in eligible:
        by_assay[summary.dominant("assay")[0]].append(summary)
    holdout: List[str] = []
    train: List[str] = []
    target_holdout = args.holdout_datasets or max(1, round(len(eligible) * args.holdout_fraction))
    for assay_group in by_assay.values():
        for position, summary in enumerate(assay_group):
            if position == 0 and len(holdout) < target_holdout and len(assay_group) > 1:
                holdout.append(summary.dataset_id)
            else:
                train.append(summary.dataset_id)
    while len(holdout) < target_holdout and len(train) > 1:
        holdout.append(train.pop())

    stats = {
        "matched_tissues": tissues,
        "num_candidate_datasets": len(summaries),
        "num_eligible_datasets": len(eligible),
        "num_rejected_datasets": len(rejected),
        "rejection_reasons": dict(Counter(reason for row in rejected for reason in row["reasons"])),
        "rejected": rejected if args.include_rejected else rejected[: args.max_rejected_reported],
    }
    return sorted(train), sorted(holdout), stats


def filter_split_csv(source: str, destination: Path, keep: set) -> Tuple[int, set]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    files: set = set()
    with open(source, "r", encoding="utf-8", newline="") as reader_handle, destination.open(
        "w", encoding="utf-8", newline=""
    ) as writer_handle:
        reader = csv.DictReader(reader_handle)
        writer = csv.DictWriter(writer_handle, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        for row in reader:
            if (row.get("dataset_id") or "unknown") in keep:
                writer.writerow(row)
                kept += 1
                files.add(row["file_path"])
    return kept, files


def write_bundle(
    source_bundle: Dict[str, Any],
    source_bundle_path: Path,
    output_root: Path,
    data_run_id: str,
    keep_datasets: set,
    split_names: Sequence[str],
) -> Dict[str, Any]:
    """Materialise a bundle restricted to ``keep_datasets``.

    The gene vocabulary and gene maps are reused verbatim so the subset stays
    input-compatible with the atlas checkpoint.
    """
    destination_root = Path(output_root) / data_run_id
    destination_root.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    used_files: set = set()
    for split in split_names:
        source = source_bundle["split_paths"].get(split)
        if source is None:
            continue
        target = destination_root / f"records_{split}.csv"
        kept, files = filter_split_csv(resolve_relative(source_bundle_path, source), target, keep_datasets)
        if kept == 0:
            target.unlink(missing_ok=True)
            continue
        split_paths[split] = str(target)
        counts[split] = kept
        used_files |= files

    payload = dict(source_bundle)
    payload["split_paths"] = split_paths
    payload["file_metas"] = [meta for meta in source_bundle["file_metas"] if meta["file_path"] in used_files]
    payload["file_gene_maps_path"] = resolve_relative(source_bundle_path, source_bundle["file_gene_maps_path"])
    payload["records"] = None
    payload["summary"] = {
        **source_bundle.get("summary", {}),
        "derived_from": str(source_bundle_path),
        "num_datasets": len(keep_datasets),
        "split_sizes": counts,
        "num_files": len(used_files),
    }
    bundle_file = destination_root / "data_bundle.slim.json"
    with bundle_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return {"data_run_id": data_run_id, "bundle_path": str(bundle_file), "split_sizes": counts, "num_files": len(used_files)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Training config; used only to resolve output.root.")
    parser.add_argument("--output-root", default=None, help="Overrides output.root from --config.")
    parser.add_argument("--source-data-run-id", default="atlas_train_2608")
    parser.add_argument("--name", default="atlas_matched_biology_v1", help="Prefix for the generated bundles.")
    parser.add_argument("--tissue", nargs="*", default=None, help="Explicit tissues; default picks the best covered.")
    parser.add_argument("--num-tissues", type=int, default=1)
    parser.add_argument("--min-dominant-fraction", type=float, default=0.8)
    parser.add_argument("--min-healthy-fraction", type=float, default=0.8)
    parser.add_argument("--age-min", type=float, default=20.0)
    parser.add_argument("--age-max", type=float, default=70.0)
    parser.add_argument("--require-age", action="store_true")
    parser.add_argument("--min-cells-per-dataset", type=int, default=20_000)
    parser.add_argument("--min-coarse-cell-types", type=int, default=3)
    parser.add_argument("--max-datasets", type=int, default=40)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--holdout-datasets", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--write-bundles", action="store_true")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--max-rejected-reported", type=int, default=50)
    parser.add_argument("--output", default=None, help="Manifest path; defaults to <output root>/<name>/manifest.json")
    return parser.parse_args(argv)


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    if not args.config:
        raise ValueError("pass --output-root or --config")
    from atlas_eval.paths import ensure_on_path

    ensure_on_path()
    from vq_pipeline.runtime import load_config

    return Path(load_config(args.config)["output"]["root"])


def build(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = resolve_output_root(args)
    source_path = bundle_path(output_root, args.source_data_run_id)
    source_bundle = read_bundle_json(source_path)
    split_paths = {
        split: resolve_relative(source_path, path)
        for split, path in (source_bundle.get("split_paths") or {}).items()
        if split in args.splits
    }
    if not split_paths:
        raise ValueError(f"{source_path} exposes no manifest CSVs for splits {args.splits}")

    summaries = summarize_datasets(split_paths)
    train_ids, holdout_ids, stats = select_datasets(summaries, args)
    manifest: Dict[str, Any] = {
        "name": args.name,
        "source_data_run_id": args.source_data_run_id,
        "source_bundle_path": str(source_path),
        "criteria": {
            key: value
            for key, value in vars(args).items()
            if key not in {"config", "output", "output_root", "write_bundles", "include_rejected"}
        },
        "selection": stats,
        "train_dataset_ids": train_ids,
        "matched_ood_dataset_ids": holdout_ids,
        "dataset_summaries": {
            dataset_id: summaries[dataset_id].to_json()
            for dataset_id in train_ids + holdout_ids
        },
    }
    if args.write_bundles:
        manifest["bundles"] = {
            "train": write_bundle(
                source_bundle, source_path, output_root, f"{args.name}_train", set(train_ids), args.splits
            ),
            "matched_ood": write_bundle(
                source_bundle, source_path, output_root, f"{args.name}_matched_ood", set(holdout_ids), args.splits
            ),
        }
    output = Path(args.output) if args.output else output_root / args.name / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    manifest["manifest_path"] = str(output)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    manifest = build(parse_args(argv))
    print(json.dumps({key: value for key, value in manifest.items() if key != "dataset_summaries"}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
