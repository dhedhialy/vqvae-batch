# VQ-VAE for scRNA-seq with Batch Effect Correction

Discrete representation learning for single-cell RNA-seq with batch-aware decoding,
adversarial batch disentanglement, and cell-type classification.

Real-data training matches scVI on batch mixing, cell-type conservation, and
cross-batch transfer — while yielding a discrete 64-code latent the baselines lack.
See `DISENTANGLE_RESULTS.md` and `PROJECT_UPDATE.md` for the full results (161k cells,
15 donors, 16 cell types).

## Quick Start

```bash
pip install -r requirements.txt

# Train with synthetic data (default: 5000 cells, 5 batches, 8 cell types)
python train.py

# Train with subsampled data (30% — fast, good for testing)
python train.py --subsample 0.3 --epochs 20

# Train with adversarial batch classifier
python train.py --use-adversary --adversary-alpha 0.5 --epochs 50
```

## Evaluate

```bash
python evaluate.py --checkpoint output/best_model.pt
```

## Train on Another Device

1. Copy the project folder to the target device:
   ```bash
   rsync -avz ./ user@other-device:~/vqvae-batch/
   ```
2. On the other device:
   ```bash
   cd ~/vqvae-batch
   pip install -r requirements.txt
   python train.py --subsample 0.3 --epochs 20
   ```

Data is generated synthetically by default — no scRNA-seq files needed.

## Real scRNA-seq Data

Train and evaluate on real `.h5ad` data with a Negative-Binomial VQ-VAE. The loader
returns **raw integer counts** (from `.raw`) — the correct input for NB
reconstruction — while selecting the same highly-variable gene set the scVI baselines use,
so metrics are directly comparable.

```bash
# Train on real data (GPU recommended; 161k-cell dataset takes ~4 min/15 epochs)
python train_real.py \
  --data-path data/cellxgene.h5ad \
  --batch-key donor_id --celltype-key cell_type \
  --n-top-genes 2000 --max-batches 15 --max-cell-types 20 \
  --use-ema --use-adversary --alpha-ramp 8 --adversary-alpha 0.5 \
  --device cuda

# Evaluate with the same metrics as the baselines (batch LISI, cell-type LISI,
# leave-one-donor-out transfer accuracy)
python eval_real.py --checkpoint output/best.pt --data-path data/cellxgene.h5ad
```

Key real-data flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-key` | `donor_id` | obs column encoding technical batches |
| `--celltype-key` | `cell_type` | obs column encoding cell types |
| `--min-cells-per-batch` | 50 | drop batches with fewer cells |
| `--n-top-genes` | 2000 | HVG selection count |
| `--use-ema` | off | EMA codebook (more stable than gradients) |
| `--restart-dead-codes` | off | resample unused codes mid-training |
| `--mmd-weight` | 0 | kernel MMD batch regularizer (**not recommended on real data** — collapses cell types; see DISENTANGLE_RESULTS.md) |
| `--code-batch-weight` | 2.0 | code↔batch independence regularizer |
| `--use-adversary` | off | DANN-style adversarial batch classifier |
| `--alpha-ramp` | 6 | epochs over which the adversary ramps up |

## Architecture

```
cell counts -> Encoder -> continuous embedding
continuous embedding -> VectorQuantizer -> discrete code
discrete code + batch embedding -> Decoder -> Negative-Binomial (mu, theta, pi)
                        |
              Adversary (optional) -> predict batch (gradient reversed)
                        |
              Classifier (optional) -> predict cell type
```

NB recon = mean negative-log NB likelihood over raw counts, with library-size scaling
from the per-cell count sum (no learned library size, per `vqvae_batch.py:308`).

## Key Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-codes` | 64 | Number of discrete VQ codes |
| `--latent-dim` | 128 | Encoder / code dimension |
| `--n-batches` | 5 | Number of batches |
| `--use-adversary` | False | Enable adversarial batch classifier |
| `--subsample` | None | Fraction of data to use (e.g. 0.3) |

## Disentanglement Framework (ID + OOD)

Use `eval_disentanglement.py` to evaluate batch/assay leakage, biological retention,
and code-usage alignment under both in-distribution and OOD splits.

```bash
python eval_disentanglement.py \
  --checkpoint output/real_vqvae/best.pt \
  --id-data-path data/in_distribution.h5ad \
  --ood-data-paths data/ood_protocol.h5ad,data/ood_tissue.h5ad \
  --ood-names protocol_ood,tissue_ood \
  --batch-fields dataset_id,assay \
  --bio-field cell_type \
  --transfer-group-field dataset_id \
  --context-field coarse_cell_type \
  --max-cells-per-split 200000 \
  --output output/disentanglement_eval.json
```

Main outputs:
- Probe metrics for leakage and retention (`logreg` + `RF`, acc and macro-F1),
- LISI per target field (batch-like and biological),
- Leave-one-group transfer for biological labels,
- Conditional code-usage TV distance by context,
- ID→OOD probe transfer on overlapping classes.

## Controlled-Biology Subset Builder

Use `build_controlled_subset.py` to create a matched-biology training subset before OOD
generalization tests.

```bash
python build_controlled_subset.py \
  --data-path data/atlas_full.h5ad \
  --output-h5ad data/atlas_matched.h5ad \
  --batch-field dataset_id \
  --celltype-field cell_type \
  --constraints tissue=lung,disease=healthy \
  --min-cells-per-batch 1000 \
  --max-batches 30 \
  --max-cells 300000
```

This writes:
- a filtered `.h5ad` subset for controlled training,
- a JSON manifest recording constraints and final batch/cell-type counts.
