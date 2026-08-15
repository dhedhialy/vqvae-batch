"""Render the atlas disentanglement scorecard JSON into a human-readable report.

Produces ``report.md`` plus per-bundle bar charts next to the input scorecard
JSON.  Pure offline: does not need the atlas server or models.

Usage:
    python make_scorecard_report.py path/to/disentanglement_scorecard.json [--manifest path/to/manifest.json] [--out-dir path]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH_VIEWS = ["input_expression", "encoder_z_e", "bio_z_q", "bio_code_onehot",
               "technical_embedding", "bio_reconstruction", "full_reconstruction"]
SHOW_VIEWS = ["input_expression", "encoder_z_e", "bio_z_q", "bio_code_onehot",
              "technical_embedding", "bio_reconstruction", "full_reconstruction"]
BUNDLE_LABELS = {
    "atlas_train_2608": "In-distribution (atlas test split)",
    "atlas_ood_unseen_protocol_2608": "OOD: unseen protocol / assay",
    "atlas_ood_unseen_tissue_2608": "OOD: unseen tissue",
    "atlas_ood_disease_2608": "OOD: disease",
}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _probe_acc(rep: Dict[str, Any], label: str) -> Any:
    probes = rep.get("batch_leakage" if label in ("dataset_id", "assay", "donor_id")
                     else "biology_conservation", {}).get("probes", {})
    entry = probes.get(label)
    return entry.get("accuracy") if entry else None


def _ilisi(rep: Dict[str, Any], label: str) -> Any:
    entry = rep.get("batch_leakage", {}).get("ilisi", {}).get(label)
    return entry.get("lisi") if entry else None


def _clisi(rep: Dict[str, Any], label: str) -> Any:
    entry = rep.get("biology_conservation", {}).get("clisi", {}).get(label)
    return entry.get("lisi") if entry else None


def _cross_dataset(rep: Dict[str, Any]) -> Any:
    cd = rep.get("biology_conservation", {}).get("cross_dataset_transfer", {})
    return cd.get("mean_accuracy") if cd else None


def _summary(rep: Dict[str, Any]) -> Dict[str, Any]:
    return rep.get("summary", {})


def build_markdown(scorecard: Dict[str, Any], manifest: Dict[str, Any] | None) -> str:
    lines: List[str] = []
    lines.append("# Atlas VQ-VAE disentanglement scorecard")
    lines.append("")
    lines.append(f"- **run**: `{scorecard.get('run_id')}`")
    ck = scorecard.get("checkpoint", {})
    lines.append(f"- **checkpoint**: `{ck.get('checkpoint_path')}` (epoch {ck.get('checkpoint_epoch')}, "
                 f"best {ck.get('checkpoint_metric')} = {_fmt(ck.get('best_metric'))})")
    lines.append(f"- **schema**: v{scorecard.get('schema_version')} | "
                 f"probe model `{scorecard.get('settings', {}).get('probe_model')}` | "
                 f"projection dim {scorecard.get('settings', {}).get('projection_dim')}")
    lines.append(f"- **sampling**: {scorecard.get('settings', {}).get('max_cells')} cells per split, "
                 f"{scorecard.get('settings', {}).get('lisi_cells')} cells for iLISI/cLISI "
                 f"(k={scorecard.get('settings', {}).get('lisi_k')}), kBET k={scorecard.get('settings', {}).get('kbet_k')}")
    lines.append("")
    thresholds = None
    for bundle in scorecard.get("bundles", {}).values():
        v = bundle.get("verdict", {})
        if v and "thresholds" in v:
            thresholds = v["thresholds"]
            break
    if thresholds:
        lines.append("**Verdict thresholds** (bio view vs `input_expression`):")
        for k, v in thresholds.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    lines.append("## Bundle summary")
    lines.append("")
    lines.append("| Bundle | Cells | Datasets | bio_z_q dataset leakage (rel.) | leakage reduced? | "
                 "biology retention (rel.) | biology preserved? | **disentangled?** | tech branch encodes dataset | tech branch free of biology |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for bname, bundle in scorecard.get("bundles", {}).items():
        v = bundle.get("verdict", {})
        views = v.get("views", {}).get("bio_z_q", {})
        tb = v.get("technical_branch", {})
        lines.append(
            f"| {BUNDLE_LABELS.get(bname, bname)} | {bundle.get('num_cells')} | {bundle.get('num_datasets')} "
            f"| {_fmt(views.get('relative_dataset_leakage'))} | {views.get('batch_leakage_reduced')} "
            f"| {_fmt(views.get('relative_biology_retention'))} | {views.get('biology_preserved')} "
            f"| **{views.get('disentangled')}** | {tb.get('encodes_dataset')} | {tb.get('free_of_biology')} |")
    lines.append("")

    for bname, bundle in scorecard.get("bundles", {}).items():
        lines.append(f"## {BUNDLE_LABELS.get(bname, bname)}")
        lines.append("")
        lines.append(f"_{bundle.get('num_cells')} cells / {bundle.get('num_datasets')} datasets "
                     f"from `{bname}`_")
        lines.append("")
        reps = bundle.get("representations", {})
        lines.append("| View | dataset probe acc | assay probe acc | dataset iLISI | "
                     "coarse CT probe acc | coarse cLISI | cross-dataset CT transfer | "
                     "dataset leak / input | biology readout / input |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for view in SHOW_VIEWS:
            rep = reps.get(view)
            if rep is None:
                continue
            sm = _summary(rep)
            lines.append(
                f"| `{view}` | {_fmt(_probe_acc(rep, 'dataset_id'))} | {_fmt(_probe_acc(rep, 'assay'))} "
                f"| {_fmt(_ilisi(rep, 'dataset_id'))} | {_fmt(_probe_acc(rep, 'coarse_cell_type'))} "
                f"| {_fmt(_clisi(rep, 'coarse_cell_type'))} | {_fmt(_cross_dataset(rep))} "
                f"| {_fmt(sm.get('dataset_leakage_ratio'))} | {_fmt(sm.get('coarse_cell_type_readout_ratio'))} |")
        lines.append("")

        vd = bundle.get("vq_diagnostics", {})
        cu = vd.get("codebook_usage", {})
        if cu:
            lines.append(f"- **codebook**: {_fmt(cu.get('mean_normalized_entropy'))} mean normalized entropy, "
                         f"{cu.get('mean_active_codes')} active codes / axis, "
                         f"{cu.get('total_dead_codes')} dead codes")
        tv = vd.get("conditional_code_usage_tv", {})
        if tv:
            lines.append(f"- **conditional code-usage TV** (dataset vs mean within "
                         f"{tv.get('num_contexts')} cell-type contexts): "
                         f"{_fmt(tv.get('moved_probability_mass'))}")
        bl = vd.get("most_batch_leaking_axes", [])
        bo = vd.get("most_biological_axes", [])
        if bl:
            lines.append(f"- **top batch-leaking axes**: {bl}")
        if bo:
            lines.append(f"- **top biological axes**: {bo}")
        per_axis = vd.get("per_axis", [])
        if per_axis:
            lines.append("")
            lines.append("| Axis | NMI dataset | NMI assay | NMI coarse CT | NMI fine CT | cond. code-use TV |")
            lines.append("|---|---|---|---|---|---|")
            for ax in per_axis:
                lines.append(f"| {ax.get('axis_id')} | {_fmt(ax.get('nmi_dataset_id'))} "
                             f"| {_fmt(ax.get('nmi_assay'))} | {_fmt(ax.get('nmi_coarse_cell_type'))} "
                             f"| {_fmt(ax.get('nmi_cell_type'))} | {_fmt(ax.get('conditional_moved_probability_mass'))} |")
        lines.append("")

    if manifest:
        lines.append("## Matched-biology training subset")
        lines.append("")
        crit = manifest.get("criteria", {})
        lines.append(f"**Criteria**: tissue `{crit.get('tissue')}`, "
                     f"dominant-fraction ≥ {crit.get('min_dominant_fraction')}, "
                     f"healthy ≥ {crit.get('min_healthy_fraction')}, "
                     f"age {crit.get('age_min')}–{crit.get('age_max')} (require_age={crit.get('require_age')}), "
                     f"≥ {crit.get('min_cells_per_dataset')} cells, "
                     f"≥ {crit.get('min_coarse_cell_types')} coarse cell types, "
                     f"holdout {crit.get('holdout_fraction')} of datasets")
        sel = manifest.get("selection", {})
        lines.append(f"**Selection**: {sel.get('num_candidate_datasets')} candidate datasets → "
                     f"{sel.get('num_eligible_datasets')} eligible, "
                     f"{sel.get('num_rejected_datasets')} rejected "
                     f"(reasons: {sel.get('rejection_reasons')})")
        lines.append(f"- train: {len(manifest.get('train_dataset_ids', []))} datasets")
        lines.append(f"- matched-OOD holdout: {len(manifest.get('matched_ood_dataset_ids', []))} datasets")
        bundles = manifest.get("bundles", {})
        for tag, info in bundles.items():
            ss = info.get("split_sizes", {})
            lines.append(f"- `{info.get('data_run_id')}`: {info.get('num_files')} files, "
                         f"train {ss.get('train')} / val {ss.get('val')} / test {ss.get('test')}")
        lines.append("")
    return "\n".join(lines)


def build_figures(scorecard: Dict[str, Any], out_dir: Path) -> List[str]:
    written: List[str] = []
    def _bar(ax, xs, vals, width, label):
        ok = [(x, v) for x, v in zip(xs, vals) if v is not None]
        if ok:
            ax.bar([x for x, _ in ok], [v for _, v in ok], width=width, label=label)

    for bname, bundle in scorecard.get("bundles", {}).items():
        reps = bundle.get("representations", {})
        views = [v for v in SHOW_VIEWS if v in reps]
        dataset = [_probe_acc(reps[v], "dataset_id") for v in views]
        assay = [_probe_acc(reps[v], "assay") for v in views]
        bio = [_probe_acc(reps[v], "coarse_cell_type") for v in views]
        x = list(range(len(views)))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        _bar(ax, [i - 0.3 for i in x], dataset, 0.25, "dataset_id probe acc")
        _bar(ax, x, assay, 0.25, "assay probe acc")
        _bar(ax, [i + 0.3 for i in x], bio, 0.25, "coarse cell type probe acc")
        ax.set_xticks(list(x))
        ax.set_xticklabels([v.replace("_", "\n") for v in views], rotation=0, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("linear probe accuracy")
        ax.set_title(f"{BUNDLE_LABELS.get(bname, bname)} — leakage vs biology per representation")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fname = out_dir / f"scorecard_{bname}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        written.append(str(fname))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scorecard", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    manifest = None
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    out_dir = args.out_dir or args.scorecard.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_markdown(scorecard, manifest)
    (out_dir / "scorecard_report.md").write_text(report, encoding="utf-8")
    figs = build_figures(scorecard, out_dir)
    print(f"wrote {out_dir / 'scorecard_report.md'}")
    for f in figs:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()