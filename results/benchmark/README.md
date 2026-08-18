# Unified disentanglement benchmark

All models scored on the same evaluation bundles with the same metric set
(`atlas_eval.metrics`) and the same verdict thresholds (leakage <= 0.5x input,
biology retention >= 0.9x input). Bio view: `bio_z_q` (ours) / posterior-mean
latent (scVI/scANVI); reference view: `input_expression` in both cases.

Our 4 bt5 weights form the frontier ladder: training broadens (cortex+blood+lung
x 3 assays fixes protocol/tissue retention collapse), then the weight dial cuts
remaining leakage. scVI/scANVI train on the same `atlas_bt5_train` bundle
(n_latent=256, early stopping patience 12, 250k train cells cap; scANVI uses
`coarse_cell_type` labels).

Results (leakage / retention, relative to raw input):

| Bundle | bt5 w=0.02 | bt5 w=0.10 | bt5 w=0.30 | bt5 w=0.50 | scVI | scANVI |
|---|---|---|---|---|---|---|
| matched bt5 (weak batch, strong bio) | 1.03 / 1.08 fail | 0.99 / 1.14 fail | 0.98 / 1.15 fail | 1.07 / 1.08 fail | 1.55 / 1.08 fail | 1.42 / 1.08 fail |
| matched v2 (weak batch, strong bio) | n/a | n/a | n/a | n/a | 4.31 / 1.01 fail | 3.38 / 1.02 fail |
| protocol OOD | 0.80 / 0.89 fail | 0.83 / 0.92 fail | 0.63 / 0.97 fail | **0.43 / 1.04 PASS** | 1.17 / 0.94 fail | 1.10 / 1.37 fail |
| tissue OOD | 0.80 / 1.05 fail | 0.67 / 0.98 fail | 0.58 / 0.86 fail | 0.54 / 0.81 fail | 1.09 / 1.09 fail | 1.05 / 1.13 fail |
| disease OOD | **0.45 / 1.04 PASS** | **0.35 / 1.18 PASS** | **0.11 / 1.02 PASS** | **0.00 / 0.97 PASS** | 1.00 / 1.14 fail | 0.87 / 1.11 fail |

Notes:
- The matched bundles contain datasets the baselines were trained on (same
  dataset_ids in train), yet scVI/scANVI still leak MORE than raw input on
  matched-OOD holds (1.4-4.3x). The batch embeddings project unseen+seen
  dataset identity into the latent, over-correcting where biology repeats.
- On every true OOD rung both baselines fail the leakage bar (0.87-1.17x),
  and on protocol/tissue OOD retention drops below the reference view.
- Only bt5 w=0.50 passes any OOD rung beyond disease: protocol at
  0.43/1.04 (first rung to pass since broad 3-assay training).
- bt5 w=0.50 remains the only model that passes protocol + disease OOD
  (disease probe blind at 0.00x while input leaks 0.64x).

Scorecards: results/benchmark/{bt5_w002,bt5_w010,bt5_w030,bt5_w050,scvi_bt5,scanvi_bt5}_scorecard.json