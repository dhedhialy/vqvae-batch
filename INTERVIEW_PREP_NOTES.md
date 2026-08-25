# Interview Prep Notes — Two Projects

> Format: **goal → method → result → limitation** for every answer.
> Drill: cover the right column / answers, recall from the prompts.

---

## PART 1 — BBB-Penetrant Peptide Pipeline (drug design paper)

### 1-LINE SUMMARY
Local Python pipeline that folds peptide sequences with ESMFold, rigid-docks them against EGFRvIII, and ranks candidates by a 5-part heuristic score (IPS) for BBB crossing + tumor targeting.

### THE PROBLEM (30s)
- GBM (glioblastoma): aggressive brain cancer, median survival <18 months.
- Blood-brain barrier (BBB) blocks ~98% of large-molecule drugs.
- **EGFRvIII**: tumor-specific EGFR deletion mutant (exons 2–13) — absent in normal tissue → ideal target.
- Drug = **chimeric peptide**: **Angiopep-2** (19-residue, crosses BBB via LRP1 receptor transcytosis) + **GE11** (dodecapeptide, binds EGFR/EGFRvIII).
- Existing tools = black-box web servers (PatchDock) → not reproducible. I made it local + deterministic + script-driven.

### CONSTRUCTS (Table I)
| Name | Modification | aa | MW |
|---|---|---|---|
| P1 Linear | baseline (Angiopep-2 + GE11) | 24 | 2792 |
| P2 Capped | N-terminal Ac / C-terminal NH2 | 24 | 2816 |
| P3 Cyclized | disulfide-looped GE11 variant | 26 | 3019 |
| P4 RetroEnantio | all D-amino acids, reversed | 24 | 2792 |
| P5 Scrambled | control (same composition, random order) | — | — |

### PIPELINE STEPS (methodology, in order)
1. **Target prep** — PDB **8UKX** (EGFRvIII cryo-EM). Isolate chain A, trim to **residues 270–420** (the cleft around the Δ6–273 deletion zone = tumor-exclusive pocket).
2. **Folding** — **local ESMFold** (single-sequence protein language model → 3D coords). No MSA needed → works on synthetic chimeras with no evolutionary history. ~2.41 s/seq, <4.2 GB RAM.
3. **Docking (rigid body)** — translate peptide & receptor so **centers of mass coincide** (offset = mean of receptor Cαs − mean of peptide Cαs), then **500 rotations at 10° increments** on 3 axes.
4. **Contact scoring** — Cα–Cα Euclidean distance matrix; a contact counts if `dist ≤ τ`, τ = **7.0 Å**. Classes by residue identity: hydrophobic (~83–89%), H-bonds (~11–16%), salt bridges (~0–2%).
5. **IPS ranking** — InterfacePriorityScore:
   `IPS = ω₁S_con + ω₂S_den + ω₃S_del + ω₄S_bbb + ω₅S_phy`, all ω = 0.20.
   - **S_con**: total Cα–Cα interface contacts
   - **S_den**: contacts ÷ peptide length (contacts/residue — corrects for length bias)
   - **S_del**: ControlDelta = contacts minus scrambled-baseline contacts
   - **S_bbb**: step-function of BBB criteria
   - **S_phy**: stability + physical property boundaries
6. **Controls / sensitivity** — 125 parameter perturbations (τ ∈ [6,8] Å, weight variants, cutoffs) across 3 scoring modes; also 5 ensemble scrambled controls on an extended-backbone fallback.

### RESULTS (key numbers to memorize)
- **P3 Cyclized = rank #1**: **55 contacts, density 2.115, IPS 0.828.**
- P4: 42 / 1.750 / 0.408 (#2). P5 scrambled: 30 / 1.250 / 0.400 (#3). P2: 38 / 1.583 / 0.305 (#4). P1: 37 / 1.542 / 0.280 (#5).
- **Rank stability CV = 0** across all 125 perturbations.
- Ensemble scrambled controls → **0 contacts, σ² = 0** (degenerate baseline; Z-score/Cohen's d undefined).
- **Physicochemical (Table III):** P3 Cyclized — MW 3019, pI 10.06, **instability index 43.63 (>40 = fail)**, GRAVY −0.42, charge +5.1 → satisfies **3 of 4** BBB filters → **"moderate" BBB likelihood** (corrected prior overestimate).

### KEY LIMITATIONS (be ready to admit)
- **Not thermodynamics** — no ΔG, no Kd, no binding velocities. Heuristic ordinal screen only.
- **Cα backbone only** — ignores side-chain rotamers & steric clashes (GE11's Trp/Tyr can clash). Future: LJ 6-12 / hard-sphere clash filter.
- **Language model can't encode non-natural chemistry** — D-amino acids (P4) and capping (P2) fold identically to P1 → near-identical scores.
- **Scrambled control artifact** — random sequence produced a collapsed/spurious fold with 30 contacts; resolved by extended-backbone control (0 contacts).
- BBB filters are statistical (charge/MW/GRAVY/instability), not physiological (no P-gp efflux, no LRP1 transit rate).

### ESSENTIAL NEXT STEPS
1. Flexible docking (HADDOCK / AutoDock CrankPep)
2. Explicit-solvent MD (≥1 µs) + MM/PBSA free energy
3. Force fields with D-amino acid / capping parameters
4. Solid-phase synthesis + SPR / BLI binding vs. EGFRvIII
5. Transwell BBB assays (bEnd.3 / hCMEC/D3)
6. All-atom steric clash filter
7. Chirality warning system in the pipeline

### LIKELY INTERVIEW QUESTIONS
- Why Cα–Cα instead of all-atom? → *Side-chain rotamers from unrefined LM folds are noise; backbone is a stable proxy for interface area.*
- Why ESMFold over AlphaFold? → *No MSA needed; chimeras have no homologs. Local = reproducible, no web server.*
- Why is P3 Cyclized better? → *Disulfide loop reduces entropy, locks GE11 conformation → more stable interface. But note the length bias → density term.*
- Why did the scrambled control score high? → *Collapsed fake fold; that's why we added the degenerate extended-backbone control.*
- Is IPS a binding affinity? → *No, ordinal heuristic for pre-filtering before expensive MD.*

---

## PART 2 — VQ/FSQ-VAE for scRNA-seq Batch Correction

### 1-LINE SUMMARY
VQ-VAE with a Negative-Binomial head and adversarial batch regularizer that matches scVI on real single-cell data while producing an interpretable **discrete** latent; upgraded to FSQ-VAE (no codebook).

### THE PROBLEM
scRNA-seq has **technical batch effects** (donor, lab, protocol) that obscure real biology. Goal: a latent representation where **batch information is removed but cell-type biology is preserved** — and ideally discrete, so each "code" is interpretable (maps to genes/cell types).

### METRICS (know cold — same ones used for scVI baselines)
| Metric | What it measures | Direction |
|---|---|---|
| **Batch LISI** | mixing of batches in latent space | higher = better; random mixing = log₂(#batches) = log₂(15) ≈ 3.91; perfect = 15 |
| **Celltype LISI** | biological conservation | lower = better |
| **Cross-batch accuracy** | train cell-type classifier on some donors, test on held-out donor | higher = better |

### ARCHITECTURE (from `vqvae_batch.py`)
```
counts → Encoder (MLP) → z (128-d)
z → VectorQuantizer → discrete code (64 codes)
code + batch-embedding + log-library-size → Decoder → NB(mu, theta, pi)
optional heads on z:
  BatchAdversary (DANN, gradient reversal) → predicts batch   [drives batch OUT]
  CellTypeClassifier → predicts cell type                      [drives biology IN]
```
- **Negative Binomial likelihood** on raw integer counts (scVI formulation): `Var = μ + μ²/θ`. Correct for RNA overdispersion — MSE is wrong for counts.
- **Library size** = sum of counts per cell (log), fed to decoder — no learned library size.
- **VQ loss** = codebook loss (encoder→code) + commitment cost (0.25). With EMA codebook, decay 0.99 + **dead-code restart** (Jukebox trick: reinit unused codes from encoder outputs).
- **Batch correction levers (3):**
  1. **DANN adversarial** — classifier predicts batch from z; gradient-reversal flips its gradients so encoder learns z that fools it. Ramped up over α-ramp epochs.
  2. **Kernel MMD** between P(z|batch) and P(z) — RBF kernel, median-heuristic bandwidth.
  3. **Code↔batch independence** — penalize divergence between each batch's code histogram and the global histogram (works on the discrete object biologists read).

### REAL-DATA RESULTS (CellxGene `0b75c598`: 161,152 cells, 33,920 genes, 15 donors, 16 types)
| Config | Batch LISI | Celltype LISI | Cross-batch acc |
|---|---|---|---|
| scVI L10 (baseline) | 3.275 | 1.390 | 0.867 |
| scVI L30 (baseline) | 3.660 | 1.423 | 0.850 |
| **VQ-VAE adv α=0.5 (final)** | **3.100** | **1.400** | **0.857 ± 0.038** |
| VQ-VAE adv α=1.0 | 3.420 | 1.513 | 0.824 |
| VQ-VAE adv α=0.3 | 2.947 | 1.368 | 0.861 |
| VQ-VAE no regularizer | 2.885 | 1.391 | 0.853 |
| VQ-VAE MMD 20–300 | 5.07–6.34 | 3.29–4.92 | 0.23–0.46 |

**Bottom line:** adv α=0.5 **matches scVI on all three axes simultaneously**, with the bonus of a discrete 64-code latent.

### KEY FINDINGS
1. **MMD does not transfer to real single-cell data.** It inflates batch LISI only by **overmixing and collapsing cell types** (celltype LISI 3.3–4.9, cross-batch halved). Excluded from final model.
2. **DANN adversarial is the only lever that works.** Low ramp, α = 0.3–0.5.
3. **α is a single trade-off dial**: 0.3 → conservation-biased, 0.5 → balanced (final), 1.0 → mixing-biased. Tunable for the downstream task.
4. **Discrete latent is interpretable**: 64 codes → code×celltype, code×batch matrices → code→gene→pathway signatures.

### FSQ-VAE v2 (`fsq_vae2.py`) — the upgrade
- **Why FSQ:** Vector quantization has learnable codebooks → needs EMA, dead-code restart, commitment loss tuning. **Finite Scalar Quantization** is fixed/analytic: `z = tanh(z) · scale; round(z)` with **straight-through estimator**; levels [8,8,8] = 512 codes. No codebook, no dead codes, no commitment term.
- Same goal: discrete latent + batch adversary (gradient reversal) + cell-type classifier head (normal gradient), NB decoder (scVI `NegativeBinomial` distribution), learned library-size encoder + batch embedding.
- Latent is a small VAE: encoder outputs mean + softplus variance → reparameterization → KL term.
- Also evaluates **code purity** (majority cell-type fraction per code) and **+Harmony post-processing** on the discrete latent.

### LIKELY INTERVIEW QUESTIONS
- Why NB instead of MSE? → *Counts are overdispersed; NB has μ and θ, Var = μ + μ²/θ. MSE assumes Gaussian, wrong for sparse counts.*
- What does gradient reversal do? → *Classifier tries to predict batch; reversal makes encoder learn the OPPOSITE of what helps the classifier → z becomes batch-invariant. α controls strength.*
- Why does MMD fail on real data? → *Overmixes — matches batch distributions by destroying cell-type clusters; on synthetic data it's fine, on real sparse, heterogeneous data it collapses biology.*
- What's the difference between batch LISI and celltype LISI? → *LISI = inverse Simpson index of label diversity in kNN neighborhood. Batch LISI ↑ = batches mixed. Celltype LISI ↓ = cells of same type stay together.*
- Why discrete? → *Interpretability: each code is a cluster of cells you can annotate with genes/pathways; scVI's continuous latent has no such units.*
- VQ vs FSQ? → *VQ learns codebook (EMA, restart, commitment). FSQ is analytic rounding with straight-through gradients — simpler, faster, no hyperparameters.*

### ATLAS-SCALE RESULTS (Aug 2026 — know these too)
- **Eval framework** `atlas_eval/`: leakage (dataset probe on bio latent vs raw input,
  pass <= 0.5x) + retention (cross-dataset cell-type transfer, pass >= 0.9x); rungs =
  in-distribution, matched-biology whole-dataset holdout, unseen protocol / tissue / disease.
- **bt5 train**: cortex+blood+lung, ~1M cells, 3 assays; w = conditional-code-usage weight.
- **Ours w=0.50**: protocol OOD 0.43/1.04 PASS; disease OOD 0.00/0.97 PASS; tissue 0.54/0.81
  fail (no healthy kidney in healthy-only train — structural); matched-OOD ~1.07 fail.
- **scVI/scANVI fail every rung**, leak up to 4.31x/3.38x input on matched rungs:
  per-dataset batch embeddings are random for held-out datasets. Ours has no
  per-dataset parameters — that's the mechanistic advantage.
- **FSQ v2**: Finite Scalar Quantization replaces the codebook (tanh->round,
  straight-through, levels [8,8,8] = 512 codes, learned library encoder).

---

## RAPID-FIRE DRILL (cover answers, recall)

1. Target protein & PDB? → EGFRvIII extracellular, **8UKX**, residues 270–420.
2. How does Angiopep-2 cross BBB? → **LRP1 receptor-mediated transcytosis**.
3. What does GE11 target? → **EGFR/EGFRvIII**.
4. Folding model? → **local ESMFold** (single-seq LM, no MSA).
5. Docking procedure? → center-of-mass alignment + **500 rotations / 10° steps**.
6. Contact cutoff? → Cα–Cα ≤ **7.0 Å**.
7. IPS components? → contacts, density, control-delta, BBB step, physical (ω=0.20 each).
8. Winner & numbers? → **P3 Cyclized: 55 contacts, 2.115 density, 0.828 IPS**.
9. Rank stability? → **CV = 0** over 125 perturbations.
10. Why "moderate" BBB? → **instability index 43.63 > 40** → 3/4 filters pass.
11. Biggest methodological limits? → not thermodynamics; backbone-only (no side chains); LM can't encode D-aa/capping; fake scrambled fold.
12. Dataset for VQ-VAE? → **CellxGene `0b75c598`, 161k cells, 15 donors, 16 types**, 2000 HVGs.
13. VQ-VAE final config? → **adversary α = 0.5**, EMA codebook, NB head, 64 codes.
14. Final numbers vs scVI? → **3.100 / 1.400 / 0.857** vs scVI 3.28–3.66 / 1.39–1.42 / 0.850–0.867.
15. Random-mixing batch LISI reference? → log₂(15) ≈ **3.91**.
16. Why not MMD on real data? → overmixes, **collapses cell types** (LISI 3.3–4.9, x-batch 0.23–0.46).
17. What dials batch vs biology? → **adversary α**: 0.3 / 0.5 / 1.0.
18. What does FSQ replace? → the **learned codebook** (analytic tanh→round, straight-through).
19. FSQ levels [8,8,8] = ? → **512 codes**.
20. Next interpretability step? → code→celltype→**marker genes/pathway enrichment** validation.
