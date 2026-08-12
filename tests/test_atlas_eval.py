"""Unit tests for the atlas scorecard metrics and the matched-biology builder.

These deliberately avoid torch and the atlas bundles so the framework can be
validated off the training server.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from atlas_eval import adversary_monitor, matched_biology, metrics, scorecard


@pytest.fixture()
def separated_batches():
    """Two batches that are perfectly separated in feature space."""
    rng = np.random.default_rng(0)
    features = np.concatenate(
        [rng.normal(-8.0, 0.2, size=(200, 4)), rng.normal(8.0, 0.2, size=(200, 4))]
    )
    labels = ["a"] * 200 + ["b"] * 200
    return features, labels


@pytest.fixture()
def mixed_batches():
    rng = np.random.default_rng(0)
    features = rng.normal(0.0, 1.0, size=(400, 4))
    labels = ["a", "b"] * 200
    return features, labels


def test_clean_and_encode_labels_collapse_unknown_spellings():
    cleaned = metrics.clean_labels(["A", "n/a", None, "NaN", "<unknown>", "b"])
    assert cleaned.tolist() == ["A", "unknown", "unknown", "unknown", "unknown", "b"]
    codes, classes = metrics.encode_labels(cleaned)
    assert classes == ["A", "b"]
    assert codes.tolist() == [0, -1, -1, -1, -1, 1]


def test_ilisi_separates_mixed_from_separated_batches(separated_batches, mixed_batches):
    separated = metrics.simpson_diversity(*separated_batches, k=20)
    mixed = metrics.simpson_diversity(*mixed_batches, k=20)
    assert separated["lisi"] < 1.2
    assert mixed["lisi"] > 1.7
    assert 0.0 <= separated["lisi_normalized"] < mixed["lisi_normalized"] <= 1.0


def test_kbet_rejects_separated_batches_and_accepts_mixed(separated_batches, mixed_batches):
    assert metrics.kbet_acceptance(*separated_batches, k=20)["acceptance_rate"] < 0.1
    assert metrics.kbet_acceptance(*mixed_batches, k=20)["acceptance_rate"] > 0.5


def test_neighbour_metrics_return_none_without_two_classes():
    features = np.zeros((50, 3))
    assert metrics.simpson_diversity(features, ["a"] * 50, k=5) is None
    assert metrics.kbet_acceptance(features, ["a"] * 50, k=5) is None


def test_label_probe_reads_separable_labels_and_reports_baselines(separated_batches):
    features, labels = separated_batches
    train = slice(0, None, 2)
    test = slice(1, None, 2)
    probe = metrics.label_probe(
        features[train], np.array(labels)[train], features[test], np.array(labels)[test]
    )
    assert probe["accuracy"] > 0.95
    assert probe["uniform_chance"] == pytest.approx(0.5)
    assert probe["majority_baseline"] == pytest.approx(0.5)
    assert metrics.leakage_ratio(probe) > 0.9


def test_label_probe_is_near_chance_on_noise(mixed_batches):
    features, labels = mixed_batches
    labels = np.array(labels)
    half = len(labels) // 2
    probe = metrics.label_probe(features[:half], labels[:half], features[half:], labels[half:])
    assert metrics.leakage_ratio(probe) < 0.3


def test_label_probe_needs_two_sufficiently_large_classes():
    features = np.random.default_rng(0).normal(size=(40, 3))
    labels = ["a"] * 39 + ["b"]
    assert metrics.label_probe(features, labels, features, labels, min_class_count=32) is None


def test_cross_group_transfer_holds_out_whole_groups():
    rng = np.random.default_rng(1)
    features, labels, groups = [], [], []
    for group in range(4):
        for cell_type, centre in (("t", -5.0), ("b", 5.0)):
            features.append(rng.normal(centre, 0.3, size=(60, 3)))
            labels += [cell_type] * 60
            groups += [f"dataset_{group}"] * 60
    result = metrics.cross_group_transfer(np.concatenate(features), labels, groups, max_folds=4)
    assert result["num_groups"] == 4
    assert result["num_folds"] == 4
    assert result["mean_accuracy"] > 0.9
    assert all(fold["unseen_label_fraction"] == 0.0 for fold in result["folds"])


def test_conditional_code_usage_tv_zero_when_datasets_agree():
    usage = np.tile(np.array([[[0.5, 0.5], [0.25, 0.75]]]), (40, 1, 1))
    target = ["d1"] * 20 + ["d2"] * 20
    context = ["ct"] * 40
    result = metrics.conditional_code_usage_tv(usage, target, context)
    assert result["moved_probability_mass"] == pytest.approx(0.0, abs=1e-9)
    assert result["num_contexts"] == 1


def test_conditional_code_usage_tv_measures_moved_mass_between_datasets():
    usage = np.zeros((40, 1, 2))
    usage[:20, 0] = [1.0, 0.0]
    usage[20:, 0] = [0.0, 1.0]
    result = metrics.conditional_code_usage_tv(usage, ["d1"] * 20 + ["d2"] * 20, ["ct"] * 40)
    assert result["moved_probability_mass"] == pytest.approx(0.5)
    assert result["per_axis_moved_probability_mass"] == pytest.approx([0.5])


def test_conditional_code_usage_tv_skips_small_and_single_dataset_groups():
    usage = np.zeros((10, 1, 2))
    usage[:, 0] = [1.0, 0.0]
    assert metrics.conditional_code_usage_tv(usage, ["d1"] * 10, ["ct"] * 10) is None
    assert metrics.conditional_code_usage_tv(
        usage, ["d1"] * 5 + ["d2"] * 5, ["ct"] * 10, min_cells=8
    ) is None


def test_codebook_usage_reports_dead_codes_and_entropy():
    indices = np.zeros((100, 2), dtype=int)
    indices[:, 1] = np.arange(100) % 4
    usage = metrics.codebook_usage(indices, codebook_size=4)
    assert usage["per_axis"][0] == {
        "axis_id": 0,
        "normalized_entropy": pytest.approx(0.0, abs=1e-6),
        "active_codes": 1,
        "dead_codes": 3,
    }
    assert usage["per_axis"][1]["normalized_entropy"] == pytest.approx(1.0, abs=1e-6)
    assert usage["total_dead_codes"] == 3


def test_axis_label_association_finds_the_leaking_axis():
    rng = np.random.default_rng(0)
    labels = ["a"] * 100 + ["b"] * 100
    indices = np.stack([rng.integers(0, 4, size=200), np.array([0] * 100 + [1] * 100)], axis=1)
    rows = metrics.axis_label_association(indices, labels)
    assert rows[1]["normalized_mutual_info"] > 0.9
    assert rows[0]["normalized_mutual_info"] < 0.2


def test_gaussian_projection_is_deterministic_and_shrinks_width():
    features = np.random.default_rng(0).normal(size=(20, 50))
    first = metrics.gaussian_projection(features, 8, seed=3)
    assert first.shape == (20, 8)
    assert np.array_equal(first, metrics.gaussian_projection(features, 8, seed=3))
    assert metrics.gaussian_projection(features, 100, seed=3).shape == features.shape


def test_adversary_monitor_flags_oscillation_but_not_steady_decay():
    oscillating = adversary_monitor.AdversaryChaseMonitor(num_classes=10, segment=2, warmup_steps=0)
    status = {}
    for step, accuracy in enumerate([0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.2, 0.2]):
        status = oscillating.update(step, accuracy)
    assert status["should_abort"]
    assert "chase" in status["reason"]

    decaying = adversary_monitor.AdversaryChaseMonitor(num_classes=10, segment=2, warmup_steps=0)
    for step, accuracy in enumerate([0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1]):
        status = decaying.update(step, accuracy)
    assert not status["should_abort"]


def test_adversary_monitor_waits_for_warmup_and_data():
    monitor = adversary_monitor.AdversaryChaseMonitor(num_classes=10, segment=2, warmup_steps=1000)
    for step, accuracy in enumerate([0.9, 0.9, 0.2, 0.2] * 3):
        status = monitor.update(step, accuracy)
    assert not status["should_abort"]
    assert status["direction_changes"] >= 4


def test_scan_history_replays_a_recorded_trace():
    history = [
        {"epoch": epoch, "train/batch_adversary_accuracy": accuracy}
        for epoch, accuracy in enumerate([0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.2, 0.2, 0.9, 0.9, 0.2, 0.2], start=1)
    ]
    report = adversary_monitor.scan_history(history, num_classes=10, segment=2)
    assert report["flagged_at_any_point"]
    assert adversary_monitor.scan_history([{"epoch": 1}], num_classes=10)["num_observations"] == 0


# --- scorecard aggregation (no torch, no bundles) ----------------------------


class _FakePack:
    """Duck-typed stand-in for RepresentationPack so scoring stays testable."""

    def __init__(self, labels, code_indices, soft_code_usage, codebook_size):
        self.representations = {}
        self.labels = labels
        self.code_indices = code_indices
        self.soft_code_usage = soft_code_usage
        self.meta = {
            "num_axes": code_indices.shape[1],
            "codebook_size": codebook_size,
            "splits": ["test"],
            "num_cells": len(code_indices),
        }


def _scorecard_args(**overrides):
    args = scorecard.parse_args(["--config", "c.yaml", "--run-id", "r"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_bundle_verdict_passes_when_batch_drops_and_biology_holds():
    representations = {
        "input_expression": {"summary": {
            "dataset_leakage_ratio": 0.8, "cross_dataset_cell_type_balanced_accuracy": 0.7}},
        "bio_z_q": {"summary": {
            "dataset_leakage_ratio": 0.2, "cross_dataset_cell_type_balanced_accuracy": 0.69}},
        "technical_embedding": {"summary": {
            "dataset_leakage_ratio": 0.95, "coarse_cell_type_readout_ratio": 0.05}},
    }
    verdict = scorecard.bundle_verdict(representations, _scorecard_args())
    checks = verdict["views"]["bio_z_q"]
    assert checks["relative_dataset_leakage"] == pytest.approx(0.25)
    assert checks["disentangled"]
    assert verdict["technical_branch"] == {"encodes_dataset": True, "free_of_biology": True}


def test_bundle_verdict_fails_when_biology_is_over_corrected():
    representations = {
        "input_expression": {"summary": {
            "dataset_leakage_ratio": 0.8, "cross_dataset_cell_type_balanced_accuracy": 0.7}},
        "bio_z_q": {"summary": {
            "dataset_leakage_ratio": 0.05, "cross_dataset_cell_type_balanced_accuracy": 0.3}},
    }
    checks = scorecard.bundle_verdict(representations, _scorecard_args())["views"]["bio_z_q"]
    assert checks["batch_leakage_reduced"]
    assert not checks["biology_preserved"]
    assert not checks["disentangled"]


def test_vq_diagnostics_ranks_batch_leaking_axes_first():
    rng = np.random.default_rng(0)
    n = 200
    dataset = ["d1"] * (n // 2) + ["d2"] * (n // 2)
    cell_type = ["t", "b"] * (n // 2)
    indices = np.stack(
        [
            np.array([0] * (n // 2) + [1] * (n // 2)),  # axis 0 tracks dataset
            np.array([0, 1] * (n // 2)),  # axis 1 tracks cell type
            rng.integers(0, 2, size=n),  # axis 2 is noise
        ],
        axis=1,
    )
    usage = np.zeros((n, 3, 2))
    usage[np.arange(n)[:, None], np.arange(3)[None, :], indices] = 1.0
    pack = _FakePack(
        {"dataset_id": np.array(dataset), "coarse_cell_type": np.array(cell_type)},
        indices,
        usage,
        codebook_size=2,
    )
    diagnostics = scorecard.vq_diagnostics(pack, _scorecard_args())
    assert diagnostics["most_batch_leaking_axes"][0] == 0
    assert diagnostics["most_biological_axes"][0] == 1
    assert diagnostics["conditional_code_usage_tv"]["per_axis_moved_probability_mass"][0] == pytest.approx(0.5)
    assert diagnostics["conditional_code_usage_tv"]["per_axis_moved_probability_mass"][1] == pytest.approx(0.0)


# --- matched-biology subset builder -----------------------------------------

FIELDS = [
    "file_path", "row_index", "dataset_id", "cell_type", "coarse_cell_type",
    "tissue", "disease", "assay", "donor_id", "age", "sex",
]


def _write_records(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_rows(dataset_id, *, tissue, disease, assay, age, n=40, cell_types=("t", "b", "nk")):
    return [
        {
            "file_path": f"/data/{dataset_id}.h5ad",
            "row_index": index,
            "dataset_id": dataset_id,
            "cell_type": cell_types[index % len(cell_types)],
            "coarse_cell_type": cell_types[index % len(cell_types)],
            "tissue": tissue,
            "disease": disease,
            "assay": assay,
            "donor_id": f"{dataset_id}_donor{index % 3}",
            "age": age,
            "sex": "female" if index % 2 else "male",
        }
        for index in range(n)
    ]


@pytest.fixture()
def source_bundle(tmp_path):
    rows = []
    rows += _dataset_rows("healthy_lung_a", tissue="lung", disease="normal", assay="10x 3' v3", age=45)
    rows += _dataset_rows("healthy_lung_b", tissue="lung", disease="normal", assay="10x 3' v3", age=52)
    rows += _dataset_rows("healthy_lung_c", tissue="lung", disease="normal", assay="10x 5' v1", age=38)
    rows += _dataset_rows("healthy_lung_d", tissue="lung", disease="normal", assay="10x 5' v1", age=61)
    rows += _dataset_rows("diseased_lung", tissue="lung", disease="COVID-19", assay="10x 3' v3", age=50)
    rows += _dataset_rows("healthy_blood", tissue="blood", disease="normal", assay="10x 3' v3", age=44)
    rows += _dataset_rows("infant_lung", tissue="lung", disease="normal", assay="10x 3' v3", age=1)
    rows += _dataset_rows("tiny_lung", tissue="lung", disease="normal", assay="10x 3' v3", age=40, n=4)

    root = tmp_path / "atlas_source"
    root.mkdir()
    _write_records(root / "records_train.csv", rows)
    _write_records(root / "records_test.csv", rows[:80])
    (root / "gene_maps.json").write_text("{}", encoding="utf-8")
    bundle = {
        "file_metas": [
            {"file_path": f"/data/{name}.h5ad", "num_cells": 40}
            for name in [
                "healthy_lung_a", "healthy_lung_b", "healthy_lung_c", "healthy_lung_d",
                "diseased_lung", "healthy_blood", "infant_lung", "tiny_lung",
            ]
        ],
        "gene_vocab": ["g1", "g2"],
        "file_gene_maps_path": str(root / "gene_maps.json"),
        "split_paths": {"train": str(root / "records_train.csv"), "test": str(root / "records_test.csv")},
        "summary": {"num_datasets": 8},
    }
    (root / "data_bundle.slim.json").write_text(json.dumps(bundle), encoding="utf-8")
    return tmp_path


def _args(tmp_path, **overrides):
    argv = [
        "--output-root", str(tmp_path),
        "--source-data-run-id", "atlas_source",
        "--name", "matched_v1",
        "--min-cells-per-dataset", "10",
        "--min-coarse-cell-types", "2",
    ]
    for key, value in overrides.items():
        argv.append(f"--{key.replace('_', '-')}")
        if value is not True:
            argv += [str(item) for item in (value if isinstance(value, list) else [value])]
    return matched_biology.parse_args(argv)


def test_matched_biology_selects_only_matched_datasets(source_bundle):
    manifest = matched_biology.build(_args(source_bundle))
    selected = set(manifest["train_dataset_ids"]) | set(manifest["matched_ood_dataset_ids"])
    assert selected == {"healthy_lung_a", "healthy_lung_b", "healthy_lung_c", "healthy_lung_d"}
    assert manifest["selection"]["matched_tissues"] == ["lung"]
    reasons = manifest["selection"]["rejection_reasons"]
    assert reasons["not_healthy_enough"] == 1  # diseased_lung
    assert reasons["tissue_not_matched"] == 1  # healthy_blood
    assert reasons["age_out_of_band"] == 1  # infant_lung
    assert reasons["too_few_cells"] == 1  # tiny_lung


def test_matched_biology_holdout_is_disjoint_and_assay_balanced(source_bundle):
    manifest = matched_biology.build(_args(source_bundle))
    train = manifest["train_dataset_ids"]
    holdout = manifest["matched_ood_dataset_ids"]
    assert holdout and not set(train) & set(holdout)
    summaries = manifest["dataset_summaries"]
    # the holdout must not smuggle in an unseen protocol: that is the separate
    # protocol-OOD rung, not the matched-biology one.
    assert {summaries[name]["dominant_assay"] for name in holdout} <= {
        summaries[name]["dominant_assay"] for name in train
    }

    balanced = matched_biology.build(_args(source_bundle, holdout_datasets=2))
    assert {summaries[name]["dominant_assay"] for name in balanced["matched_ood_dataset_ids"]} == {
        "10x 3' v3",
        "10x 5' v1",
    }


def test_matched_biology_respects_an_explicit_tissue_and_holdout_size(source_bundle):
    manifest = matched_biology.build(_args(source_bundle, tissue="blood", holdout_datasets=0, min_healthy_fraction=0.0))
    assert manifest["selection"]["matched_tissues"] == ["blood"]
    assert manifest["train_dataset_ids"] == ["healthy_blood"]


def test_matched_biology_writes_usable_filtered_bundles(source_bundle):
    manifest = matched_biology.build(_args(source_bundle, write_bundles=True))
    train_bundle = json.loads(Path(manifest["bundles"]["train"]["bundle_path"]).read_text(encoding="utf-8"))
    train_ids = set(manifest["train_dataset_ids"])

    assert set(train_bundle["split_paths"]) <= {"train", "test"}
    assert train_bundle["gene_vocab"] == ["g1", "g2"]
    assert len(train_bundle["file_metas"]) == len(train_ids)
    with open(train_bundle["split_paths"]["train"], encoding="utf-8") as handle:
        written = {row["dataset_id"] for row in csv.DictReader(handle)}
    assert written == train_ids
    assert Path(manifest["manifest_path"]).exists()
