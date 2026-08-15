"""Disentanglement metrics shared by every representation in the scorecard.

Three families are implemented:

``A`` batch leakage      -- label probes, iLISI, kBET acceptance
``B`` biology conservation -- cLISI, cross-dataset cell-type transfer
``C`` VQ-specific        -- per-axis leakage, conditional code-use TV, codebook health

Everything is numpy/scikit-learn only so it can be unit tested without torch,
a GPU, or the atlas bundles.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

UNKNOWN_TOKENS = {"", "na", "n/a", "nan", "none", "unknown", "<unknown>"}


def clean_labels(values: Sequence[Any]) -> np.ndarray:
    out = []
    for value in values:
        text = str("" if value is None else value).strip()
        out.append("unknown" if text.lower() in UNKNOWN_TOKENS else text)
    return np.asarray(out, dtype=object)


def known_mask(labels: np.ndarray) -> np.ndarray:
    return labels != "unknown"


def encode_labels(labels: Sequence[Any]) -> Tuple[np.ndarray, List[str]]:
    cleaned = clean_labels(labels)
    classes = sorted(set(cleaned[known_mask(cleaned)].tolist()))
    lookup = {label: idx for idx, label in enumerate(classes)}
    codes = np.asarray([lookup.get(label, -1) for label in cleaned], dtype=np.int64)
    return codes, classes


def subsample_indices(n: int, max_samples: Optional[int], seed: int) -> np.ndarray:
    if max_samples is None or n <= int(max_samples):
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_samples), replace=False))


def gaussian_projection(features: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """Fixed, label-independent linear projection that bounds neighbour-search cost."""
    if dim <= 0 or features.shape[1] <= dim:
        return np.asarray(features, dtype=np.float32)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(features.shape[1], dim)).astype(np.float32)
    return np.asarray(features, dtype=np.float32) @ matrix


def _neighbour_labels(features: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = max(1, min(int(k), len(codes) - 1))
    finder = NearestNeighbors(n_neighbors=k + 1).fit(features)
    _, neighbours = finder.kneighbors(features)
    return codes[neighbours[:, 1:]]


def simpson_diversity(
    features: np.ndarray,
    labels: Sequence[Any],
    k: int = 90,
    max_cells: Optional[int] = 20_000,
    seed: int = 1701,
) -> Optional[Dict[str, float]]:
    """Local inverse-Simpson index (LISI) of ``labels`` in a kNN neighbourhood.

    Returns the raw LISI plus a 0-1 normalisation ``(lisi - 1) / (classes - 1)``
    so batch mixing (higher is better) and cell-type purity (lower is better)
    are comparable across bundles with different class counts.
    """
    codes, classes = encode_labels(labels)
    keep = np.flatnonzero(codes >= 0)
    if len(classes) < 2 or len(keep) < k + 2:
        return None
    keep = keep[subsample_indices(len(keep), max_cells, seed)]
    codes = codes[keep]
    neighbour_codes = _neighbour_labels(np.asarray(features, dtype=np.float32)[keep], codes, k)
    counts = np.zeros((len(codes), len(classes)), dtype=np.float32)
    for class_id in range(len(classes)):
        counts[:, class_id] = (neighbour_codes == class_id).sum(axis=1)
    proportions = counts / np.clip(counts.sum(axis=1, keepdims=True), 1.0, None)
    lisi = 1.0 / np.clip((proportions ** 2).sum(axis=1), 1e-12, None)
    return {
        "lisi": float(lisi.mean()),
        "lisi_normalized": float((lisi.mean() - 1.0) / max(len(classes) - 1, 1)),
        "num_classes": int(len(classes)),
        "num_cells": int(len(codes)),
        "k": int(max(1, min(int(k), len(codes) - 1))),
    }


def kbet_acceptance(
    features: np.ndarray,
    labels: Sequence[Any],
    k: int = 40,
    alpha: float = 0.05,
    max_cells: Optional[int] = 10_000,
    seed: int = 1701,
) -> Optional[Dict[str, float]]:
    """Fraction of neighbourhoods whose batch mix matches the global mix (higher is better)."""
    from scipy import stats

    codes, classes = encode_labels(labels)
    keep = np.flatnonzero(codes >= 0)
    if len(classes) < 2 or len(keep) < k + 2:
        return None
    keep = keep[subsample_indices(len(keep), max_cells, seed)]
    codes = codes[keep]
    neighbour_codes = _neighbour_labels(np.asarray(features, dtype=np.float32)[keep], codes, k)
    k = neighbour_codes.shape[1]
    global_proportion = np.bincount(codes, minlength=len(classes)).astype(np.float64) / len(codes)
    expected = global_proportion * k
    represented = expected > 0
    observed = np.zeros((len(codes), len(classes)), dtype=np.float64)
    for class_id in range(len(classes)):
        observed[:, class_id] = (neighbour_codes == class_id).sum(axis=1)
    statistic = ((observed[:, represented] - expected[represented]) ** 2 / expected[represented]).sum(axis=1)
    p_values = stats.chi2.sf(statistic, df=int(represented.sum()) - 1)
    return {
        "acceptance_rate": float((p_values >= alpha).mean()),
        "rejection_rate": float((p_values < alpha).mean()),
        "num_classes": int(len(classes)),
        "num_cells": int(len(codes)),
        "k": int(k),
    }


def _fit_classifier(x: np.ndarray, y: np.ndarray, seed: int, model: str):
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier

    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ).fit(x, y)
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4, max_iter=10, tol=None,
        random_state=seed, n_jobs=1, average=True,
    ).fit(x, y)


def _balanced_sample(y: np.ndarray, candidates: np.ndarray, max_per_class: Optional[int], seed: int) -> np.ndarray:
    if not max_per_class or max_per_class <= 0:
        return candidates
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(y[candidates].tolist())):
        idx = candidates[y[candidates] == label]
        if len(idx) > int(max_per_class):
            idx = rng.choice(idx, int(max_per_class), replace=False)
        selected.append(idx)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def label_probe(
    train_features: np.ndarray,
    train_labels: Sequence[Any],
    test_features: np.ndarray,
    test_labels: Sequence[Any],
    *,
    min_class_count: int = 32,
    max_train_per_class: Optional[int] = 1024,
    seed: int = 1701,
    model: str = "linear",
) -> Optional[Dict[str, Any]]:
    """Supervised readout of one label from one representation.

    Reports accuracy against two baselines so a number can be read without the
    class distribution at hand: ``majority_baseline`` (predict the most frequent
    training class) and ``uniform_chance`` (1 / classes).
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    y_train = clean_labels(train_labels)
    y_test = clean_labels(test_labels)
    counts = Counter(y_train[known_mask(y_train)].tolist())
    classes = sorted(label for label, count in counts.items() if count >= int(min_class_count))
    if len(classes) < 2:
        return None
    train_idx = np.flatnonzero(np.isin(y_train, classes))
    test_idx = np.flatnonzero(np.isin(y_test, classes))
    if len(train_idx) < 2 or len(test_idx) < 2:
        return None
    train_idx = _balanced_sample(y_train, train_idx, max_train_per_class, seed)

    scaler = StandardScaler().fit(train_features[train_idx])
    classifier = _fit_classifier(
        scaler.transform(train_features[train_idx]), y_train[train_idx].astype(str), seed, model
    )
    predicted = classifier.predict(scaler.transform(test_features[test_idx]))
    gold = y_test[test_idx].astype(str)
    majority = max(counts.items(), key=lambda item: item[1])[0]
    return {
        "model": model,
        "accuracy": float(accuracy_score(gold, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro")),
        "majority_baseline": float((gold == majority).mean()),
        "uniform_chance": float(1.0 / len(classes)),
        "num_classes": int(len(classes)),
        "num_train_cells": int(len(train_idx)),
        "num_test_cells": int(len(test_idx)),
    }


def cross_group_transfer(
    features: np.ndarray,
    labels: Sequence[Any],
    groups: Sequence[Any],
    *,
    max_folds: int = 5,
    min_class_count: int = 32,
    max_train_per_class: Optional[int] = 1024,
    seed: int = 1701,
    model: str = "linear",
) -> Optional[Dict[str, Any]]:
    """Leave-group-out label transfer (groups are datasets, not donors, at atlas scale).

    Held-out group cells whose label never appeared in the fitting groups are
    reported as ``unseen_label_fraction`` rather than silently scored.
    """
    y = clean_labels(labels)
    group_values = clean_labels(groups)
    usable = known_mask(y) & known_mask(group_values)
    unique_groups = sorted(set(group_values[usable].tolist()))
    if len(unique_groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    order = list(unique_groups)
    rng.shuffle(order)
    folds = [order[idx::min(int(max_folds), len(order))] for idx in range(min(int(max_folds), len(order)))]

    fold_rows: List[Dict[str, Any]] = []
    for held_out in folds:
        test_mask = usable & np.isin(group_values, held_out)
        train_mask = usable & ~np.isin(group_values, held_out)
        if test_mask.sum() < 2 or train_mask.sum() < 2:
            continue
        probe = label_probe(
            features[train_mask], y[train_mask], features[test_mask], y[test_mask],
            min_class_count=min_class_count, max_train_per_class=max_train_per_class,
            seed=seed, model=model,
        )
        if probe is None:
            continue
        seen = np.isin(y[test_mask], sorted(set(y[train_mask].tolist())))
        probe["held_out_groups"] = list(held_out)
        probe["unseen_label_fraction"] = float(1.0 - seen.mean())
        fold_rows.append(probe)
    if not fold_rows:
        return None
    return {
        "mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
        "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "mean_majority_baseline": float(np.mean([row["majority_baseline"] for row in fold_rows])),
        "num_folds": int(len(fold_rows)),
        "num_groups": int(len(unique_groups)),
        "folds": fold_rows,
    }


def axis_label_association(indices: np.ndarray, labels: Sequence[Any]) -> Optional[List[Dict[str, Any]]]:
    """Normalised mutual information between each axis' code choice and a label."""
    from sklearn.metrics import normalized_mutual_info_score

    codes, classes = encode_labels(labels)
    valid = codes >= 0
    if len(classes) < 2 or valid.sum() < 2:
        return None
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[:, None]
    return [
        {
            "axis_id": int(axis_id),
            "normalized_mutual_info": float(
                normalized_mutual_info_score(codes[valid], indices[valid, axis_id])
            ),
        }
        for axis_id in range(indices.shape[1])
    ]


def conditional_code_usage_tv(
    soft_usage: np.ndarray,
    target: Sequence[Any],
    context: Sequence[Any],
    *,
    min_cells: int = 8,
) -> Optional[Dict[str, Any]]:
    """Held-out form of the training-time conditional code-usage alignment loss.

    ``soft_usage`` is ``cells x axes x codes``.  Within every context group
    (coarse cell type) represented by at least two targets (datasets), each
    target's equally weighted code table is compared with the group mean by the
    fraction of probability mass that would have to move (total variation).
    """
    soft_usage = np.asarray(soft_usage, dtype=np.float64)
    if soft_usage.ndim != 3:
        raise ValueError(f"soft_usage must be cells x axes x codes, got {soft_usage.shape}")
    target_labels = clean_labels(target)
    context_labels = clean_labels(context)
    valid = known_mask(target_labels) & known_mask(context_labels)
    num_axes = soft_usage.shape[1]
    per_context: List[Dict[str, Any]] = []
    per_axis_totals = np.zeros(num_axes, dtype=np.float64)
    for context_value in sorted(set(context_labels[valid].tolist())):
        in_context = valid & (context_labels == context_value)
        tables = []
        used_targets = []
        for target_value in sorted(set(target_labels[in_context].tolist())):
            rows = in_context & (target_labels == target_value)
            if rows.sum() < int(min_cells):
                continue
            tables.append(soft_usage[rows].mean(axis=0))
            used_targets.append(target_value)
        if len(tables) < 2:
            continue
        stacked = np.stack(tables, axis=0)
        mean_table = stacked.mean(axis=0, keepdims=True)
        moved_mass = 0.5 * np.abs(stacked - mean_table).sum(axis=-1)  # targets x axes
        per_axis = moved_mass.mean(axis=0)
        per_axis_totals += per_axis
        per_context.append(
            {
                "context": context_value,
                "num_targets": int(len(used_targets)),
                "moved_probability_mass": float(per_axis.mean()),
            }
        )
    if not per_context:
        return None
    return {
        "moved_probability_mass": float(np.mean([row["moved_probability_mass"] for row in per_context])),
        "per_axis_moved_probability_mass": (per_axis_totals / len(per_context)).tolist(),
        "num_contexts": int(len(per_context)),
        "min_cells": int(min_cells),
        "contexts": per_context,
    }


def codebook_usage(indices: np.ndarray, codebook_size: int) -> Dict[str, Any]:
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[:, None]
    per_axis = []
    for axis_id in range(indices.shape[1]):
        counts = np.bincount(indices[:, axis_id], minlength=int(codebook_size)).astype(np.float64)
        frequency = counts / max(counts.sum(), 1.0)
        entropy = -float((frequency * np.log(frequency + 1e-12)).sum() / math.log(int(codebook_size)))
        per_axis.append(
            {
                "axis_id": int(axis_id),
                "normalized_entropy": entropy,
                "active_codes": int((counts > 0).sum()),
                "dead_codes": int(int(codebook_size) - (counts > 0).sum()),
            }
        )
    return {
        "mean_normalized_entropy": float(np.mean([row["normalized_entropy"] for row in per_axis])),
        "mean_active_codes": float(np.mean([row["active_codes"] for row in per_axis])),
        "total_dead_codes": int(sum(row["dead_codes"] for row in per_axis)),
        "codebook_size": int(codebook_size),
        "per_axis": per_axis,
    }


def leakage_ratio(probe: Optional[Dict[str, Any]]) -> Optional[float]:
    """Probe accuracy above its majority baseline, on a 0-1 headroom scale.

    0 means the representation is no better than always predicting the most
    frequent class; 1 means the label is perfectly readable.
    """
    if probe is None:
        return None
    headroom = 1.0 - probe["majority_baseline"]
    if headroom <= 1e-9:
        return 0.0
    return float(max(0.0, probe["accuracy"] - probe["majority_baseline"]) / headroom)


__all__ = [
    "axis_label_association",
    "clean_labels",
    "codebook_usage",
    "conditional_code_usage_tv",
    "cross_group_transfer",
    "encode_labels",
    "gaussian_projection",
    "kbet_acceptance",
    "known_mask",
    "label_probe",
    "leakage_ratio",
    "simpson_diversity",
    "subsample_indices",
]
