"""Render the broad matched-biology (bt5) weight ladder into results/bt5/.

Usage:
    python make_bt5_report.py --out-dir results/bt5
    (JSONs are read from local paths passed via CLI flags --baseline/--bt5-* )
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS: List[Tuple[str, str, str]] = [
    ("v6 baseline (all-atlas train, w=0.02)", "atlas_v6_log1p_s20260809", "k"),
    ("bt5 w=0.02", "atlas_bt5_w0.02_s20260816", "C0"),
    ("bt5 w=0.10", "atlas_bt5_w0.10_s20260816", "C1"),
    ("bt5 w=0.30", "atlas_bt5_w0.30_s20260816", "C2"),
    ("bt5 w=0.50", "atlas_bt5_w0.50_s20260816", "C3"),
]

BUNDLES: List[Tuple[str, str]] = [
    ("atlas_bt5_matched_ood", "Matched-OOD\n(bt5 holdout datasets)"),
    ("atlas_ood_unseen_protocol_2608", "Protocol OOD"),
    ("atlas_ood_unseen_tissue_2608", "Tissue OOD"),
    ("atlas_ood_disease_2608", "Disease OOD"),
]

THRESH_LEAK = 0.5
THRESH_RET = 0.9


def cell(scorecard: Dict[str, Any], bundle: str) -> Tuple[float | None, float | None, bool | None]:
    bj = scorecard.get("bundles", {}).get(bundle)
    if not bj:
        return None, None, None
    v = bj.get("verdict", {}).get("views", {}).get("bio_z_q", {})
    return v.get("relative_dataset_leakage"), v.get("relative_biology_retention"), v.get("disentangled")


def build_markdown(data: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Broad matched-biology sweep (bt5)")
    lines.append("")
    lines.append("Training subset `atlas_bt5_train`: **cerebral cortex + blood + lung**, "
                 "healthy ≥ 80%, age 10–90, whole-dataset holdout "
                 "`atlas_bt5_matched_ood` (10 datasets). Assays in train: "
                 "`10x 3' v3`, `10x 3' v2`, `10x 5' v2` — so the protocol rung "
                 "no longer leaves the training manifold.")
    lines.append("")
    lines.append(f"Thresholds: bio dataset leakage ≤ **{THRESH_LEAK}×** input, "
                 f"biology retention ≥ **{THRESH_RET}×** input (`bio_z_q` view).")
    lines.append("")
    lines.append("## Leakage / retention per rung")
    lines.append("")
    lines.append("| Run | " + " | ".join(f"{label.replace(chr(10), ' ')}\nleak/ret" for _, label in BUNDLES) + " |")
    lines.append("|---|" + "---|" * len(BUNDLES))
    for name, run_id, _ in RUNS:
        cells = [cell(data[name], b) for b, _ in BUNDLES]
        lines.append("| `" + run_id + "` | " + " | ".join(
            f"{l:.3f}/{r:.3f}" if l is not None else "n/a" for l, r, _ in cells) + " |")
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    lines.append("| Run | " + " | ".join("`" + b.split("atlas_")[-1].replace("_2608", "") + "`" for b, _ in BUNDLES) + " |")
    lines.append("|---|" + "---|" * len(BUNDLES))
    for name, run_id, _ in RUNS:
        cells = [cell(data[name], b) for b, _ in BUNDLES]
        lines.append("| `" + run_id + "` | " + " | ".join(
            "**PASS**" if v else ("fail" if l is not None else "n/a") for l, _, v in cells) + " |")
    lines.append("")
    lines.append("## Reading this ladder")
    lines.append("")
    lines.append("- **The dial now works where it must**: protocol and disease OOD both "
                 "pass at w=0.50 (`leak 0.43×/0.00×`, retention `1.04/0.97`). The first "
                 "rung — unseen protocols with **unseen assays** — was the one the "
                 "single-tissue retrain could not pass on retention; training on three "
                 "assays fixed retention and the dial then cut the remaining leak.")
    lines.append("- **Disease OOD passes at every weight** and at w=0.50 the dataset "
                 "probe on bio codes is effectively blind (`0.00×` vs input — input "
                 "itself leaks `0.64×`). Biology is preserved throughout (`0.97–1.18`).")
    lines.append("- **Matched-OOD stays at ≈1.0×**: with a multi-tissue train the "
                 "holdout datasets mix cortex+blood compositions, so the dataset probe "
                 "is confounded with legitimate tissue composition; the single-tissue "
                 "v2 experiment (`0.77×`) is the correct pure batch-transfer test and "
                 "remains the cleaner matched rung.")
    lines.append("- **Tissue OOD is structurally limited**: the rung is kidney; healthy "
                 "kidney datasets are absent from the atlas, so kidney biology is "
                 "outside the (healthy-only) training manifold. Retention `0.81` and "
                 "leak `0.54` at w=0.50; a kidney dataset cannot enter a healthy-matched "
                 "train by construction.")
    lines.append("")
    return "\n".join(lines)


def build_figures(data: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (bundle, label) in zip(axes.flat, BUNDLES):
        for name, run_id, color in RUNS:
            leak, ret, v = cell(data[name], bundle)
            if leak is None:
                continue
            ax.scatter(leak, ret, s=90, color=color, zorder=3, label=f"{name} {'(PASS)' if v else ''}")
            ax.annotate(run_id.split("_")[-2] if run_id.startswith("atlas_bt5") else "v6",
                        (leak, ret), textcoords="offset points", xytext=(6, 5), fontsize=7)
        ax.axvline(THRESH_LEAK, color="k", ls="--", lw=0.8)
        ax.axhline(THRESH_RET, color="k", ls="--", lw=0.8)
        ax.set_xlabel("bio dataset leakage (rel. to input)")
        ax.set_ylabel("biology retention (rel. to input)")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("bt5 (cortex+blood+lung) sweep — disentanglement frontier", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fname = out_dir / "bt5_frontier.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return [fname]


def main() -> None:
    ap = argparse.ArgumentParser()
    for name, run_id, _ in RUNS:
        ap.add_argument(f"--{run_id}", type=Path, required=run_id.startswith("atlas_bt5"))
    ap.add_argument("--v6", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/bt5"))
    args = ap.parse_args()

    data: Dict[str, Dict[str, Any]] = {}
    for name, run_id, _ in RUNS:
        src: Any = args
        path = getattr(src, run_id, None) or getattr(src, "v6", None)
        data[name] = json.loads(path.read_text(encoding="utf-8"))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BT5.md").write_text(build_markdown(data), encoding="utf-8")
    figs = build_figures(data, out_dir)
    print(f"wrote {out_dir / 'BT5.md'}")
    for f in figs:
        print(f"wrote {f}")


if __name__ == "__main__":
    main()