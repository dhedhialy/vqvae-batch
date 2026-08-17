"""Unified disentanglement benchmark across settings and models.

The benchmark is a fixed matrix: every *setting* (a train bundle plus a set of
evaluation bundles) is scored for every *model* (our VQ-VAE runs, scVI, scANVI)
with the identical metric set and verdict thresholds, and the results land in
one table.  Settings cover the ladder the single experiments were probing one
at a time, plus the hard case the user asked for: batches that are technically
weak while biology is strong, where over-correction is the failure mode.

Settings
--------

- ``matched``          whole-dataset holdout of a biology-matched train subset
                       (weak technical batch, same biology both sides)
- ``matched_bt5``      same on the broad multi-tissue subset (3 assays)
- ``disease_ood``      unseen disease datasets (strong biology shifts)
- ``protocol_ood``     unseen protocol / assay datasets
- ``tissue_ood``       unseen tissue datasets

A model row is a trained checkpoint plus the scorecard JSONs of that checkpoint
scored on every evaluation bundle.  The orchestrator here only aggregates JSONs
that already exist on disk (produced by ``atlas_eval.scorecard`` for the VQ-VAE
and ``atlas_eval.baselines`` for scVI/scANVI); it never trains anything itself.
This keeps every cell reproducible by a single command and makes the matrix
grow by adding files, not code.

    python -m atlas_eval.benchmark \
        --root results/benchmark \
        --model "ours bt5 w=0.50" --scorecard path/to/bt5_w050.json \
        --model "scVI" --scorecard path/to/scvi_bt5.json \
        --model "scANVI" --scorecard path/to/scanvi_bt5.json \
        --output results/benchmark/BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BUNDLE_ORDER = [
    "atlas_bt5_matched_ood",
    "atlas_matched_biology_v2_matched_ood",
    "atlas_ood_unseen_protocol_2608",
    "atlas_ood_unseen_tissue_2608",
    "atlas_ood_disease_2608",
]

BUNDLE_LABELS = {
    "atlas_bt5_matched_ood": "matched bt5\n(weak batch,\nstrong biology)",
    "atlas_matched_biology_v2_matched_ood": "matched v2\n(weak batch,\nstrong biology)",
    "atlas_ood_unseen_protocol_2608": "protocol OOD",
    "atlas_ood_unseen_tissue_2608": "tissue OOD",
    "atlas_ood_disease_2608": "disease OOD",
}

BIO_VIEWS = {"bio_z_q": "ours", "latent": "scVI/scANVI"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "SCORECARD_JSON"),
                        help="Repeatable: model label + scorecard JSON path.")
    parser.add_argument("--output", default="BENCHMARK.md")
    return parser.parse_args(argv)


def _view_name(scorecard: Dict[str, Any]) -> Optional[str]:
    for view in ("bio_z_q", "latent"):
        if view in scorecard.get("representations", {}):
            return view
    return None


def _cell(scorecard: Dict[str, Any], bundle_id: str) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    bundle = scorecard.get("bundles", {}).get(bundle_id)
    if not bundle:
        return None, None, None
    view = _view_name(bundle)
    if view is None:
        return None, None, None
    verdict = bundle.get("verdict", {}).get("views", {}).get(view, {})
    return (
        verdict.get("relative_dataset_leakage"),
        verdict.get("relative_biology_retention"),
        verdict.get("disentangled"),
    )


def build_markdown(models: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines: List[str] = []
    lines.append("# Unified disentanglement benchmark")
    lines.append("")
    lines.append("Every row is one model scored on the same evaluation bundles with the "
                 "same metric set (`atlas_eval.metrics`) and the same verdict thresholds "
                 "(leakage ≤ 0.5× input, biology retention ≥ 0.9× input). "
                 "The bio view is `bio_z_q` for our VQ-VAE and the posterior-mean latent "
                 "for scVI/scANVI; the reference view is `input_expression` in both cases.")
    lines.append("")

    for label, scorecard in models:
        lines.append(f"- **{label}** — `{scorecard.get('run_id')}` "
                     f"({scorecard.get('model_type', 'vqvae')}, schema "
                     f"v{scorecard.get('schema_version')})")
    lines.append("")

    lines.append("| Model | " + " | ".join(BUNDLE_LABELS[b] for b in BUNDLE_ORDER) + " |")
    lines.append("|---|" + "---|" * len(BUNDLE_ORDER))
    for label, scorecard in models:
        cells = [_cell(scorecard, b) for b in BUNDLE_ORDER]
        rendered = []
        for leak, ret, verdict in cells:
            if leak is None:
                rendered.append("n/a")
                continue
            mark = "**PASS**" if verdict else "fail"
            rendered.append(f"{leak:.2f}/{ret:.2f} {mark}")
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    lines.append("")

    lines.append("Format: `leak / retention verdict` — leak is the relative dataset "
                 "leakage of the bio view vs raw input, retention the relative "
                 "cross-dataset cell-type transfer vs raw input.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.model:
        raise SystemExit("pass at least one --model NAME SCORECARD_JSON")

    models: List[Tuple[str, Dict[str, Any]]] = []
    for label, path in args.model:
        scorecard = json.loads(Path(path).read_text(encoding="utf-8"))
        models.append((label, scorecard))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(models), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()