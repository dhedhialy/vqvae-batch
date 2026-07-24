# VQ-VAE for scRNA-seq with Batch Effect Correction

Discrete representation learning for single-cell RNA-seq with batch-aware decoding, adversarial batch disentanglement, and cell-type classification.

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

To use real data (coming soon), place your AnnData file in `data/` and use:

```python
from data import load_real_data
X, batches, cell_types = load_real_data("data/my_data.h5ad")
```

## Architecture

```
cell counts -> Encoder -> continuous embedding
continuous embedding -> VectorQuantizer -> discrete code
discrete code + batch embedding -> Decoder -> reconstructed counts
                        |
              Adversary (optional) -> predict batch (gradient reversed)
                        |
              Classifier (optional) -> predict cell type
```

## Key Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-codes` | 64 | Number of discrete VQ codes |
| `--latent-dim` | 128 | Encoder / code dimension |
| `--n-batches` | 5 | Number of batches |
| `--use-adversary` | False | Enable adversarial batch classifier |
| `--subsample` | None | Fraction of data to use (e.g. 0.3) |
