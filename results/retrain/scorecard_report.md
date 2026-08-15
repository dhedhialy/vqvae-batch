# Atlas VQ-VAE disentanglement scorecard

- **run**: `atlas_matched_bio_v2_s20260815`
- **checkpoint**: `/stor/znx/vq_2608_runs/atlas_matched_bio_v2_s20260815/checkpoints/best.pt` (epoch 12, best recon_loss = 0.194)
- **schema**: v1.0 | probe model `linear` | projection dim 256
- **sampling**: 40000 cells per split, 20000 cells for iLISI/cLISI (k=90), kBET k=40

**Verdict thresholds** (bio view vs `input_expression`):
- `max_relative_dataset_leakage`: 0.5
- `min_relative_biology_retention`: 0.9
- `min_technical_branch_dataset_leakage`: 0.8
- `max_technical_branch_biology_leakage`: 0.2

## Bundle summary

| Bundle | Cells | Datasets | bio_z_q dataset leakage (rel.) | leakage reduced? | biology retention (rel.) | biology preserved? | **disentangled?** | tech branch encodes dataset | tech branch free of biology |
|---|---|---|---|---|---|---|---|---|---|
| atlas_matched_biology_v2_matched_ood | 40000 | 8 | 0.773 | False | 0.986 | True | **False** | False | True |
| OOD: unseen protocol / assay | 40000 | 12 | 0.734 | False | 0.813 | False | **False** | False | True |
| OOD: unseen tissue | 40000 | 9 | 0.730 | False | 0.811 | False | **False** | False | True |
| OOD: disease | 40000 | 12 | 0.312 | True | 0.934 | True | **True** | False | True |

## atlas_matched_biology_v2_matched_ood

_40000 cells / 8 datasets from `atlas_matched_biology_v2_matched_ood`_

| View | dataset probe acc | assay probe acc | dataset iLISI | coarse CT probe acc | coarse cLISI | cross-dataset CT transfer | dataset leak / input | biology readout / input |
|---|---|---|---|---|---|---|---|
| `input_expression` | 0.505 | - | 4.068 | 0.996 | 1.049 | 0.994 | 0.156 | 0.970 |
| `encoder_z_e` | 0.668 | - | 2.460 | 1.000 | 1.025 | 1.000 | 0.434 | 0.999 |
| `bio_z_q` | 0.484 | - | 3.531 | 0.999 | 1.005 | 1.000 | 0.121 | 0.995 |
| `bio_code_onehot` | 0.482 | - | 3.504 | 0.999 | 1.005 | 1.000 | 0.117 | 0.994 |
| `technical_embedding` | 0.104 | - | 1.000 | 0.867 | 1.000 | 0.346 | 0.000 | 0.000 |
| `bio_reconstruction` | 0.479 | - | 3.353 | 0.999 | 1.008 | 1.000 | 0.113 | 0.993 |
| `full_reconstruction` | 0.476 | - | 3.361 | 0.999 | 1.008 | 1.000 | 0.107 | 0.995 |

- **codebook**: 0.978 mean normalized entropy, 16.0 active codes / axis, 0 dead codes
- **conditional code-usage TV** (dataset vs mean within 4 cell-type contexts): 0.057
- **top batch-leaking axes**: [3, 23, 6, 30, 28]
- **top biological axes**: [26, 4, 21, 29, 11]

| Axis | NMI dataset | NMI assay | NMI coarse CT | NMI fine CT | cond. code-use TV |
|---|---|---|---|---|---|
| 0 | 0.081 | - | 0.246 | 0.237 | 0.045 |
| 1 | 0.095 | - | 0.252 | 0.246 | 0.040 |
| 2 | 0.084 | - | 0.249 | 0.240 | 0.038 |
| 3 | 0.082 | - | 0.142 | 0.158 | 0.082 |
| 4 | 0.087 | - | 0.260 | 0.250 | 0.062 |
| 5 | 0.085 | - | 0.215 | 0.207 | 0.084 |
| 6 | 0.069 | - | 0.167 | 0.234 | 0.095 |
| 7 | 0.088 | - | 0.258 | 0.248 | 0.058 |
| 8 | 0.090 | - | 0.235 | 0.250 | 0.058 |
| 9 | 0.089 | - | 0.253 | 0.242 | 0.036 |
| 10 | 0.090 | - | 0.220 | 0.227 | 0.053 |
| 11 | 0.084 | - | 0.255 | 0.245 | 0.055 |
| 12 | 0.088 | - | 0.223 | 0.253 | 0.073 |
| 13 | 0.076 | - | 0.211 | 0.212 | 0.092 |
| 14 | 0.080 | - | 0.195 | 0.192 | 0.053 |
| 15 | 0.089 | - | 0.260 | 0.249 | 0.045 |
| 16 | 0.100 | - | 0.241 | 0.237 | 0.056 |
| 17 | 0.091 | - | 0.217 | 0.231 | 0.051 |
| 18 | 0.090 | - | 0.257 | 0.246 | 0.043 |
| 19 | 0.093 | - | 0.258 | 0.248 | 0.065 |
| 20 | 0.088 | - | 0.241 | 0.231 | 0.044 |
| 21 | 0.083 | - | 0.255 | 0.244 | 0.034 |
| 22 | 0.090 | - | 0.256 | 0.245 | 0.051 |
| 23 | 0.080 | - | 0.171 | 0.186 | 0.095 |
| 24 | 0.084 | - | 0.217 | 0.212 | 0.054 |
| 25 | 0.093 | - | 0.256 | 0.246 | 0.055 |
| 26 | 0.081 | - | 0.256 | 0.245 | 0.049 |
| 27 | 0.089 | - | 0.257 | 0.247 | 0.061 |
| 28 | 0.102 | - | 0.215 | 0.231 | 0.050 |
| 29 | 0.088 | - | 0.259 | 0.249 | 0.040 |
| 30 | 0.084 | - | 0.194 | 0.249 | 0.075 |
| 31 | 0.086 | - | 0.251 | 0.241 | 0.043 |

## OOD: unseen protocol / assay

_40000 cells / 12 datasets from `atlas_ood_unseen_protocol_2608`_

| View | dataset probe acc | assay probe acc | dataset iLISI | coarse CT probe acc | coarse cLISI | cross-dataset CT transfer | dataset leak / input | biology readout / input |
|---|---|---|---|---|---|---|---|
| `input_expression` | 0.849 | 0.754 | 1.773 | 0.631 | 2.178 | 0.438 | 0.681 | 0.182 |
| `encoder_z_e` | 0.880 | 0.807 | 1.549 | 0.696 | 1.925 | 0.508 | 0.747 | 0.327 |
| `bio_z_q` | 0.763 | 0.674 | 1.661 | 0.564 | 2.254 | 0.421 | 0.500 | 0.034 |
| `bio_code_onehot` | 0.763 | 0.677 | 1.670 | 0.571 | 2.285 | 0.400 | 0.499 | 0.049 |
| `technical_embedding` | 0.526 | 0.161 | 1.000 | 0.087 | 1.442 | 0.255 | 0.000 | 0.000 |
| `bio_reconstruction` | 0.738 | 0.655 | 1.860 | 0.543 | 2.329 | 0.372 | 0.448 | 0.000 |
| `full_reconstruction` | 0.739 | 0.653 | 1.839 | 0.534 | 2.312 | 0.364 | 0.449 | 0.000 |

- **codebook**: 0.485 mean normalized entropy, 16.0 active codes / axis, 0 dead codes
- **conditional code-usage TV** (dataset vs mean within 9 cell-type contexts): 0.192
- **top batch-leaking axes**: [11, 12, 4, 5, 27]
- **top biological axes**: [15, 9, 28, 19, 30]

| Axis | NMI dataset | NMI assay | NMI coarse CT | NMI fine CT | cond. code-use TV |
|---|---|---|---|---|---|
| 0 | 0.124 | 0.100 | 0.088 | 0.201 | 0.218 |
| 1 | 0.130 | 0.109 | 0.095 | 0.152 | 0.125 |
| 2 | 0.156 | 0.144 | 0.113 | 0.157 | 0.179 |
| 3 | 0.143 | 0.130 | 0.115 | 0.160 | 0.176 |
| 4 | 0.167 | 0.159 | 0.092 | 0.198 | 0.190 |
| 5 | 0.146 | 0.137 | 0.075 | 0.137 | 0.174 |
| 6 | 0.140 | 0.135 | 0.087 | 0.174 | 0.229 |
| 7 | 0.150 | 0.145 | 0.098 | 0.140 | 0.195 |
| 8 | 0.120 | 0.108 | 0.085 | 0.160 | 0.185 |
| 9 | 0.132 | 0.102 | 0.129 | 0.246 | 0.225 |
| 10 | 0.133 | 0.117 | 0.095 | 0.130 | 0.182 |
| 11 | 0.178 | 0.169 | 0.068 | 0.152 | 0.198 |
| 12 | 0.158 | 0.157 | 0.079 | 0.135 | 0.257 |
| 13 | 0.117 | 0.105 | 0.064 | 0.090 | 0.190 |
| 14 | 0.118 | 0.109 | 0.073 | 0.146 | 0.099 |
| 15 | 0.185 | 0.162 | 0.187 | 0.289 | 0.214 |
| 16 | 0.152 | 0.135 | 0.105 | 0.132 | 0.181 |
| 17 | 0.127 | 0.119 | 0.084 | 0.131 | 0.189 |
| 18 | 0.149 | 0.145 | 0.110 | 0.173 | 0.203 |
| 19 | 0.104 | 0.102 | 0.086 | 0.234 | 0.224 |
| 20 | 0.123 | 0.108 | 0.098 | 0.142 | 0.157 |
| 21 | 0.145 | 0.129 | 0.089 | 0.199 | 0.238 |
| 22 | 0.122 | 0.116 | 0.075 | 0.101 | 0.178 |
| 23 | 0.109 | 0.107 | 0.064 | 0.148 | 0.229 |
| 24 | 0.121 | 0.112 | 0.091 | 0.146 | 0.234 |
| 25 | 0.135 | 0.134 | 0.082 | 0.104 | 0.164 |
| 26 | 0.140 | 0.138 | 0.075 | 0.158 | 0.196 |
| 27 | 0.166 | 0.165 | 0.097 | 0.143 | 0.202 |
| 28 | 0.118 | 0.100 | 0.101 | 0.167 | 0.180 |
| 29 | 0.149 | 0.138 | 0.107 | 0.204 | 0.178 |
| 30 | 0.128 | 0.120 | 0.111 | 0.197 | 0.136 |
| 31 | 0.129 | 0.127 | 0.068 | 0.161 | 0.210 |

## OOD: unseen tissue

_40000 cells / 9 datasets from `atlas_ood_unseen_tissue_2608`_

| View | dataset probe acc | assay probe acc | dataset iLISI | coarse CT probe acc | coarse cLISI | cross-dataset CT transfer | dataset leak / input | biology readout / input |
|---|---|---|---|---|---|---|---|
| `input_expression` | 0.811 | 0.849 | 1.944 | 0.742 | 2.426 | 0.590 | 0.725 | 0.583 |
| `encoder_z_e` | 0.829 | 0.858 | 1.625 | 0.779 | 1.834 | 0.640 | 0.753 | 0.643 |
| `bio_z_q` | 0.676 | 0.757 | 1.769 | 0.654 | 2.010 | 0.517 | 0.529 | 0.442 |
| `bio_code_onehot` | 0.676 | 0.761 | 1.755 | 0.653 | 2.025 | 0.511 | 0.530 | 0.440 |
| `technical_embedding` | 0.311 | 0.306 | 1.000 | 0.292 | 2.759 | 0.313 | 0.000 | 0.000 |
| `bio_reconstruction` | 0.677 | 0.761 | 1.820 | 0.637 | 2.136 | 0.519 | 0.532 | 0.414 |
| `full_reconstruction` | 0.673 | 0.753 | 1.821 | 0.640 | 2.128 | 0.518 | 0.526 | 0.420 |

- **codebook**: 0.459 mean normalized entropy, 13.375 active codes / axis, 84 dead codes
- **conditional code-usage TV** (dataset vs mean within 6 cell-type contexts): 0.216
- **top batch-leaking axes**: [13, 12, 7, 4, 25]
- **top biological axes**: [15, 21, 8, 9, 1]

| Axis | NMI dataset | NMI assay | NMI coarse CT | NMI fine CT | cond. code-use TV |
|---|---|---|---|---|---|
| 0 | 0.259 | 0.184 | 0.175 | 0.163 | 0.242 |
| 1 | 0.085 | 0.067 | 0.062 | 0.052 | 0.093 |
| 2 | 0.204 | 0.113 | 0.117 | 0.114 | 0.223 |
| 3 | 0.237 | 0.155 | 0.122 | 0.113 | 0.250 |
| 4 | 0.367 | 0.263 | 0.209 | 0.187 | 0.304 |
| 5 | 0.235 | 0.224 | 0.175 | 0.159 | 0.196 |
| 6 | 0.179 | 0.086 | 0.088 | 0.095 | 0.288 |
| 7 | 0.310 | 0.192 | 0.127 | 0.120 | 0.299 |
| 8 | 0.091 | 0.096 | 0.098 | 0.096 | 0.147 |
| 9 | 0.252 | 0.197 | 0.247 | 0.210 | 0.205 |
| 10 | 0.225 | 0.118 | 0.097 | 0.099 | 0.184 |
| 11 | 0.269 | 0.207 | 0.144 | 0.136 | 0.239 |
| 12 | 0.282 | 0.119 | 0.074 | 0.083 | 0.331 |
| 13 | 0.391 | 0.193 | 0.101 | 0.118 | 0.330 |
| 14 | 0.199 | 0.140 | 0.133 | 0.142 | 0.097 |
| 15 | 0.242 | 0.247 | 0.295 | 0.245 | 0.216 |
| 16 | 0.136 | 0.059 | 0.065 | 0.065 | 0.181 |
| 17 | 0.173 | 0.126 | 0.116 | 0.131 | 0.171 |
| 18 | 0.105 | 0.032 | 0.072 | 0.087 | 0.209 |
| 19 | 0.174 | 0.063 | 0.088 | 0.089 | 0.213 |
| 20 | 0.104 | 0.074 | 0.053 | 0.051 | 0.159 |
| 21 | 0.100 | 0.109 | 0.112 | 0.087 | 0.133 |
| 22 | 0.206 | 0.097 | 0.077 | 0.091 | 0.270 |
| 23 | 0.154 | 0.114 | 0.124 | 0.146 | 0.204 |
| 24 | 0.169 | 0.089 | 0.082 | 0.098 | 0.287 |
| 25 | 0.254 | 0.139 | 0.098 | 0.102 | 0.204 |
| 26 | 0.259 | 0.170 | 0.155 | 0.149 | 0.238 |
| 27 | 0.174 | 0.080 | 0.069 | 0.077 | 0.258 |
| 28 | 0.053 | 0.027 | 0.023 | 0.021 | 0.107 |
| 29 | 0.205 | 0.109 | 0.103 | 0.127 | 0.203 |
| 30 | 0.259 | 0.148 | 0.125 | 0.125 | 0.161 |
| 31 | 0.230 | 0.155 | 0.116 | 0.124 | 0.266 |

## OOD: disease

_40000 cells / 12 datasets from `atlas_ood_disease_2608`_

| View | dataset probe acc | assay probe acc | dataset iLISI | coarse CT probe acc | coarse cLISI | cross-dataset CT transfer | dataset leak / input | biology readout / input |
|---|---|---|---|---|---|---|---|
| `input_expression` | 0.842 | 0.820 | 2.357 | 0.871 | 1.212 | 0.697 | 0.677 | 0.312 |
| `encoder_z_e` | 0.805 | 0.806 | 1.799 | 0.910 | 1.199 | 0.688 | 0.603 | 0.519 |
| `bio_z_q` | 0.613 | 0.618 | 2.023 | 0.878 | 1.213 | 0.651 | 0.212 | 0.351 |
| `bio_code_onehot` | 0.617 | 0.622 | 2.032 | 0.881 | 1.218 | 0.657 | 0.219 | 0.365 |
| `technical_embedding` | 0.232 | 0.207 | 1.528 | 0.050 | 1.642 | 0.217 | 0.000 | 0.000 |
| `bio_reconstruction` | 0.617 | 0.608 | 2.046 | 0.882 | 1.251 | 0.647 | 0.219 | 0.372 |
| `full_reconstruction` | 0.609 | 0.607 | 2.041 | 0.882 | 1.250 | 0.649 | 0.204 | 0.370 |

- **codebook**: 0.429 mean normalized entropy, 15.90625 active codes / axis, 3 dead codes
- **conditional code-usage TV** (dataset vs mean within 9 cell-type contexts): 0.187
- **top batch-leaking axes**: [23, 24, 17, 18, 31]
- **top biological axes**: [28, 21, 1, 20, 3]

| Axis | NMI dataset | NMI assay | NMI coarse CT | NMI fine CT | cond. code-use TV |
|---|---|---|---|---|---|
| 0 | 0.209 | 0.120 | 0.256 | 0.176 | 0.203 |
| 1 | 0.236 | 0.146 | 0.362 | 0.163 | 0.080 |
| 2 | 0.148 | 0.071 | 0.170 | 0.142 | 0.178 |
| 3 | 0.225 | 0.157 | 0.317 | 0.145 | 0.159 |
| 4 | 0.323 | 0.187 | 0.368 | 0.214 | 0.266 |
| 5 | 0.295 | 0.167 | 0.334 | 0.196 | 0.200 |
| 6 | 0.176 | 0.072 | 0.191 | 0.150 | 0.245 |
| 7 | 0.212 | 0.098 | 0.252 | 0.158 | 0.229 |
| 8 | 0.180 | 0.113 | 0.205 | 0.155 | 0.153 |
| 9 | 0.212 | 0.145 | 0.265 | 0.197 | 0.165 |
| 10 | 0.222 | 0.142 | 0.294 | 0.171 | 0.173 |
| 11 | 0.281 | 0.160 | 0.322 | 0.179 | 0.223 |
| 12 | 0.182 | 0.088 | 0.193 | 0.136 | 0.255 |
| 13 | 0.289 | 0.146 | 0.330 | 0.179 | 0.262 |
| 14 | 0.169 | 0.090 | 0.181 | 0.161 | 0.099 |
| 15 | 0.249 | 0.170 | 0.320 | 0.217 | 0.163 |
| 16 | 0.137 | 0.072 | 0.181 | 0.106 | 0.148 |
| 17 | 0.174 | 0.083 | 0.178 | 0.158 | 0.179 |
| 18 | 0.124 | 0.051 | 0.129 | 0.139 | 0.226 |
| 19 | 0.230 | 0.139 | 0.278 | 0.191 | 0.169 |
| 20 | 0.268 | 0.162 | 0.392 | 0.184 | 0.141 |
| 21 | 0.299 | 0.195 | 0.433 | 0.187 | 0.129 |
| 22 | 0.184 | 0.084 | 0.219 | 0.139 | 0.244 |
| 23 | 0.156 | 0.078 | 0.151 | 0.164 | 0.200 |
| 24 | 0.139 | 0.061 | 0.140 | 0.120 | 0.230 |
| 25 | 0.208 | 0.116 | 0.241 | 0.151 | 0.177 |
| 26 | 0.264 | 0.143 | 0.315 | 0.192 | 0.219 |
| 27 | 0.150 | 0.057 | 0.181 | 0.127 | 0.216 |
| 28 | 0.249 | 0.147 | 0.391 | 0.151 | 0.109 |
| 29 | 0.210 | 0.108 | 0.235 | 0.207 | 0.182 |
| 30 | 0.232 | 0.142 | 0.297 | 0.194 | 0.137 |
| 31 | 0.183 | 0.089 | 0.194 | 0.148 | 0.227 |

## Matched-biology training subset

**Criteria**: tissue `['cerebral cortex']`, dominant-fraction ≥ 0.6, healthy ≥ 0.8, age 10.0–90.0 (require_age=False), ≥ 10000 cells, ≥ 3 coarse cell types, holdout 0.25 of datasets
**Selection**: 210 candidate datasets → 34 eligible, 176 rejected (reasons: {'tissue_not_matched': 176, 'tissue_not_homogeneous': 19, 'not_healthy_enough': 54, 'age_out_of_band': 17, 'too_few_cell_types': 22, 'too_few_cells': 23})
- train: 26 datasets
- matched-OOD holdout: 8 datasets
- `atlas_matched_biology_v2_train`: 26 files, train 697323 / val 154527 / test 168975
- `atlas_matched_biology_v2_matched_ood`: 8 files, train 228740 / val 20839 / test 34671
