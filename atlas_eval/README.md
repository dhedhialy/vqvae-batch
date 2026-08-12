# atlas_eval — disentanglement scorecard for the CELLxGENE atlas VQ-VAE

The donor-scale prototype in this repository (`eval_real.py`, `metrics.py`) answers
"is batch information gone and is biology kept?" for 15 donors and one codebook.
`atlas_eval` promotes that idea to the atlas model `atlas_v6_log1p_s20260809`:
210 datasets, 32 axes × 16 codes, a technical (dataset/assay) decoder branch,
and three existing OOD bundles.

Everything is driven off the trained checkpoint plus a data bundle; no
preprocessing is reimplemented — the modules import the `vq_2608` loader, model
builder and runtime directly.

## Modules

| Module | What it does | Needs torch / server |
|---|---|---|
| `metrics.py` | probes, iLISI/cLISI, kBET, cross-dataset transfer, per-axis NMI, conditional code-use TV, codebook health | no |
| `representations.py` | encodes a bundle into every scored view of the same cells | yes |
| `scorecard.py` | orchestration + JSON scorecard + verdicts | only to extract |
| `matched_biology.py` | builds the biology-matched training subset and its matched-OOD holdout | no |
| `adversary_monitor.py` | early-warning for the encoder/adversary chase that killed the v6 adversary | no |

## 1. Scorecard

```bash
source /stor/znx/vq_2608/scripts/env.server.sh && cd /stor/znx/vq_2608
python -m atlas_eval.scorecard \
    --config configs/v6_weak_supervision.yaml \
    --run-id atlas_v6_log1p_s20260809
```

Seven views of the *same* sampled cells are scored under identical labels:
`input_expression` (leakage upper bound), `encoder_z_e`, `bio_z_q`,
`bio_code_onehot`, `technical_embedding`, `bio_reconstruction`,
`full_reconstruction`. High-dimensional views pass through one fixed,
label-independent Gaussian projection, so linear readability survives while
kNN searches stay affordable.

Per view and per label the JSON records:

* **batch leakage** — `dataset_id` / `assay` / `donor_id` probes (accuracy plus
  majority and uniform baselines), iLISI, kBET acceptance
* **biology conservation** — `coarse_cell_type` / `cell_type` / `tissue` /
  `disease` probes, atlas-fitted transfer probes, cLISI, leave-dataset-out
  cell-type transfer
* **VQ diagnostics** — per-axis NMI with dataset/assay/cell type, per-axis
  conditional code-use TV (the eval twin of the training loss), code entropy and
  dead codes, plus the ranked most-batch-leaking and most-biological axes
* **verdict** — bio-view leakage and biology retention *relative to
  `input_expression`*, checked against documented thresholds, and whether the
  technical branch behaves as designed (reads dataset, not cell type)

Bundles: atlas `test` split, then `atlas_ood_unseen_protocol_2608`,
`atlas_ood_unseen_tissue_2608`, `atlas_ood_disease_2608` — same metrics, same
sampling, so the OOD ladder is directly comparable.

## 2. Matched-biology subset

```bash
python -m atlas_eval.matched_biology \
    --config configs/v6_weak_supervision.yaml \
    --source-data-run-id atlas_train_2608 \
    --name atlas_matched_biology_v1 --write-bundles
```

Keeps datasets that share tissue, are predominantly healthy, sit in one age
band and carry enough cell types, then holds out whole datasets — balanced
across assays so the holdout is *not* an accidental protocol shift. The
manifest records the criteria, the per-dataset biology summary and the
rejection reasons; `--write-bundles` materialises `<name>_train` and
`<name>_matched_ood` bundles that `src/train.py --data-run-id` consumes with the
atlas gene vocabulary unchanged.

Retrain on `<name>_train`, then run the same scorecard on the matched-OOD rung
first and the protocol/tissue/disease rungs after. Bio codes that mix unseen
*matched* datasets while keeping cell-type transfer mean the technical factors
transferred; bio codes that look clean in-distribution but leak on the matched
holdout mean dataset-specific quirks were fitted instead.

## 3. Adversary chase monitor

If the adversary is re-enabled (`batch_adversary_weight > 0`), wire the monitor
into the training loop so an oscillating run aborts in minutes instead of hours:

```python
from atlas_eval.adversary_monitor import AdversaryChaseMonitor

monitor = AdversaryChaseMonitor(num_classes=len(model.nuisance_categories["dataset_id"]))
...
status = monitor.update(step, float(outputs["batch_adversary_accuracy"]))
if status["should_abort"]:
    raise RuntimeError(status["reason"])
```

Recorded runs can be checked after the fact with
`python -m atlas_eval.adversary_monitor --metrics <run>/metrics.json --num-classes 210`.

## Tests

`pytest tests/test_atlas_eval.py` — metric behaviour, scorecard aggregation and
the subset builder are covered with synthetic data, so the framework can be
validated without the atlas server.
