# Unified disentanglement benchmark

Every row is one model scored on the same evaluation bundles with the same metric set (`atlas_eval.metrics`) and the same verdict thresholds (leakage ≤ 0.5× input, biology retention ≥ 0.9× input). The bio view is `bio_z_q` for our VQ-VAE and the posterior-mean latent for scVI/scANVI; the reference view is `input_expression` in both cases.

- **ours v6 baseline** — `atlas_v6_log1p_s20260809` (vqvae, schema v1.0)
- **ours bt5 w=0.02** — `atlas_bt5_w0.02_s20260816` (vqvae, schema v1.0)
- **ours bt5 w=0.10** — `atlas_bt5_w0.10_s20260816` (vqvae, schema v1.0)
- **ours bt5 w=0.30** — `atlas_bt5_w0.30_s20260816` (vqvae, schema v1.0)
- **ours bt5 w=0.50** — `atlas_bt5_w0.50_s20260816` (vqvae, schema v1.0)
- **scVI (bt5)** — `scvi_bt5_s20260816` (scvi, schema v1.0)
- **scANVI (bt5)** — `scanvi_bt5_s20260816` (scanvi, schema v1.0)

| Model | matched bt5
(weak batch,
strong biology) | matched v2
(weak batch,
strong biology) | protocol OOD | tissue OOD | disease OOD |
|---|---|---|---|---|---|
| ours v6 baseline | n/a | 1.24/1.04 fail | 0.84/0.99 fail | 0.91/1.07 fail | 0.59/1.14 fail |
| ours bt5 w=0.02 | 1.03/1.08 fail | 0.86/1.06 fail | 0.80/0.89 fail | 0.80/1.05 fail | 0.45/1.04 **PASS** |
| ours bt5 w=0.10 | 0.99/1.14 fail | 0.72/1.04 fail | 0.83/0.92 fail | 0.67/0.98 fail | 0.35/1.18 **PASS** |
| ours bt5 w=0.30 | 0.98/1.15 fail | 0.92/0.99 fail | 0.63/0.97 fail | 0.58/0.86 fail | 0.11/1.02 **PASS** |
| ours bt5 w=0.50 | 1.07/1.08 fail | 1.26/0.95 fail | 0.43/1.04 **PASS** | 0.54/0.81 fail | 0.00/0.97 **PASS** |
| scVI (bt5) | 1.55/1.08 fail | 4.31/1.01 fail | 1.17/0.94 fail | 1.09/1.09 fail | 1.00/1.14 fail |
| scANVI (bt5) | 1.42/1.08 fail | 3.38/1.02 fail | 1.10/1.37 fail | 1.05/1.13 fail | 0.87/1.11 fail |

Format: `leak / retention verdict` — leak is the relative dataset leakage of the bio view vs raw input, retention the relative cross-dataset cell-type transfer vs raw input.
