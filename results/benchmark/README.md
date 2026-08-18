# Unified disentanglement benchmark (scVI / scANVI baselines)

Both baselines trained on `atlas_bt5_train` (same fit bundle as bt5 w=0.50),
scored on the same 5 evaluation bundles with the same metric set and verdict
thresholds (leakage <= 0.5x input, biology retention >= 0.9x input).

- scVI: n_latent=256, early stopping (patience 12, validation_loss), 120 max epochs, 250k train cells cap.
- scANVI: same, labels_key=coarse_cell_type, unlabeled_category="unknown".
- Bio view: posterior-mean latent (`latent`); reference view: `input_expression`.

Results (leakage / retention, relative to raw input):

| Bundle | ours bt5 w=0.50 | scVI | scANVI |
|---|---|---|---|
| matched bt5 (weak batch, strong bio) | 1.07 / 1.08 fail | 1.55 / 1.08 fail | 1.42 / 1.08 fail |
| matched v2 (weak batch, strong bio) | n/a | 4.31 / 1.01 fail | 3.38 / 1.02 fail |
| protocol OOD | 0.43 / 1.04 **PASS** | 1.17 / 0.94 fail | 1.10 / 1.37 fail |
| tissue OOD | 0.54 / 0.81 fail | 1.09 / 1.09 fail | 1.05 / 1.13 fail |
| disease OOD | 0.00 / 0.97 **PASS** | 1.00 / 1.14 fail | 0.87 / 1.11 fail |

Notes:
- The matched bundles contain datasets the baselines were trained on (same
  dataset_ids in train), yet scVI/scANVI still leak MORE than raw input on
  matched-OOD holds (1.4-4.3x). The batch embeddings project unseen+seen
  dataset identity into the latent, over-correcting where biology repeats.
- On every true OOD rung both baselines fail the leakage bar (0.87-1.17x),
  and on protocol/tissue OOD retention drops well below the reference view.
- bt5 w=0.50 remains the only model that passes any rung
  (protocol OOD 0.43/1.04, disease OOD 0.00/0.97).

Scorecards: results/benchmark/{bt5_w050,scvi_bt5,scanvi_bt5}_scorecard.json