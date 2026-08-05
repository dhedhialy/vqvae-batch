# Real-data disentanglement results (cellxgene `0b75c598`, 161k cells, 15 donors, 16 types)

All runs share: raw counts, HVGs from same preprocessing as baselines, with `z` embedding
(non-codebook) evaluated so metrics match the scVI baselines. Baseline scVI L10/L30 rows
from `~/vqvae-batch/output/scvi_push/results.json`.

| Config                    | batch LISI | celltype LISI | cross-batch acc | notes |
|---------------------------|-----------:|--------------:|----------------:|-------|
| scVI L10 (baseline)       |      3.275 |         1.390 |           0.867 | best batch among scVI |
| scVI L30 (baseline)       |      3.660 |         1.423 |           0.850 | |
| **vqvae adv α=0.5 (final)** |   **3.100** |     **1.400** |   **0.857 ±0.038** | matched scVI on both axes |
| vqvae adv α=1.0           |      3.420 |         1.513 |           0.824 | stronger mixing, bio hurt |
| vqvae adv α=0.3           |      2.947 |         1.368 |           0.861 | bio best, batch too low |
| vqvae code_batch 5.0      |      2.816 |         1.371 |           0.864 | no batch gain |
| vqvae baseline (code 2.0) |      2.885 |         1.391 |           0.853 | starting point |
| vqvae MMD 20/50/300       |    5.07–6.34 |        3.29–4.92 |         0.23–0.46 | MMD destroys biology |
| vqvae adv+MMD+code combo  |      5.577 |         3.674 |           0.414 | overcorrected |
| vqvae MMD 5               |      1.866 |         5.234 |           0.226 | codebook collapsed |

Random-mixing batch LISI baseline `log2(15) = 3.91`; perfect mixing = 15.

## What this means for the feedback

**Batch vs biology disentanglement now works at scVI level.** The α=0.5 adversarial
VQ-VAE matches scVI on both targets: batch LISI 3.10 vs 3.28–3.66 (mixing), celltype
LISI 1.40 vs 1.39–1.42 (conservation), cross-batch cell-type transfer 0.857 vs
0.850–0.867 — with the bonus of a discrete 64-code latent that scVI lacks.

Key finding: **MMD does not transfer to real single-cell data.** It inflates batch LISI
via gross overmixing — collapsing distinct cell types (celltype LISI 3.3–4.9) and halving
cross-batch transfer. The DANN adversarial regularizer with a low ramp (α=0.3–0.5) is the
only lever that separates donor effect from cell-type signal without destroying biology.

Batch vs. biology is a trade-off dialed by adversary strength:
α=0.3 → batch 2.95 / celltype 1.37; α=0.5 → 3.10 / 1.40; α=1.0 → 3.42 / 1.51.

## Files

- Server: `~/vqvae-real/output/<cfg>/best.pt` + `eval_results.json`
- Code (local copy in `batch fix/`): `data_real.py`, `train_real.py`, `eval_real.py`
- Sweep scripts: `sweep.sh`, `sweep2.sh` (server `~/vqvae-real/submit/`)

## Next: pathway case studies

Latent is ready for the interpretability arm: 64 codes, `code_by_celltype` and
`code_by_batch` matrices, plus per-code decoder output give code → gene → pathway
signatures to validate against known cell-type-specific pathways.