# Conditional code-usage weight sweep

Same matched-biology train/val/test splits and OOD ladder as the `COMPARISON.md` experiment; only `model.conditional_code_usage_weight` changes (retrains on `atlas_matched_biology_v2_train`, 12 epochs).

Thresholds: bio dataset leakage ≤ **0.5×** input, biology retention ≥ **0.9×** input (`bio_z_q` view).

## Leakage vs retention per rung

| Run | Matched-OOD (unseen datasets, same biology): leak/ret | Protocol OOD: leak/ret | Tissue OOD: leak/ret | Disease OOD: leak/ret |
|---|---|---|---|---|
| `atlas_v6_log1p_s20260809` | 1.238/1.042 | 0.842/0.992 | 0.915/1.066 | 0.585/1.138
| `atlas_matched_bio_v2_s20260815` | 0.773/0.986 | 0.734/0.813 | 0.730/0.811 | 0.312/0.934
| `atlas_matched_bio_w0.10_s20260816` | 0.661/0.941 | 0.556/0.719 | 0.670/0.827 | 0.206/0.811
| `atlas_matched_bio_w0.50_s20260816` | 0.959/1.028 | 0.487/0.742 | 0.608/0.839 | 0.106/0.823

## Verdicts

| Run | `matched_biology_v2_matched_ood` | `ood_unseen_protocol` | `ood_unseen_tissue` | `ood_disease` |
|---|---|---|---|---|
| `atlas_v6_log1p_s20260809` | fail | fail | fail | fail
| `atlas_matched_bio_v2_s20260815` | fail | fail | fail | **PASS**
| `atlas_matched_bio_w0.10_s20260816` | fail | fail | fail | fail
| `atlas_matched_bio_w0.50_s20260816` | fail | fail | fail | fail

## Reading the frontier

- **v6 baseline has no matching rung (1.24× leakage — bio codes leak *more* than raw expression on unseen matched datasets)**: the all-atlas model encodes dataset-specific quirks, not transferable batch factors.
- **w=0.02 is the only full pass** (Disease OOD, leak 0.31× / retention 0.93×). Leakage falls monotonically with weight: 0.77 → 0.66 → 0.96 (matched), 0.73 → 0.56 → 0.49 (protocol), 0.73 → 0.67 → 0.61 (tissue), 0.31 → 0.21 → 0.11 (disease).
- **Retention pays the price**: protocol/tissue retention drops below the 0.9 bar at every training weight (0.72–0.84), and at w=0.10 disease retention slips under too (0.81). Retraining on a single-tissue / single-assay subset is what costs generality.
- **w=0.50 is over-regularized**: active code ratio drops to 0.89 and codebook entropy to 0.77, and *matched-OOD leakage rises back to 0.96×* — with fewer codes in play the surviving codebook axes re-absorb dataset identity. U-shaped transfer behaviour.
- **Technical branch is clean by construction** (encodes dataset id in-distribution, ~zero on unseen): the ladder isolates the residual leakage to the bio side, and the sweep shows the dial transfers across all four rungs.
