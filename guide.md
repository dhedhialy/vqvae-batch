# A Practical Guide to Single-Cell RNA-seq, Batch Effects, and Our VQ-VAE Project

Jul 25, 2026, 12:00 AM

## Why this project matters

Our goal is to build a VQ-VAE for single-cell RNA-seq data that has both strong representation ability and interpretability.

There is usually a trade-off. Topic models are relatively easy to interpret because each topic can be connected to a group of genes or a biological program, but their representations may be too simple for difficult downstream tasks. Large neural networks and single-cell foundation models can learn much stronger representations, but it is often hard to explain what each dimension or prediction means.

We want a model that performs well on tasks such as cell-type classification while also giving us discrete codes that can be connected to genes, pathways, cell types, and cell states.

The first major problem we need to solve is batch effect. If the model mainly learns which experiment a cell came from, then even a high classification score may not mean that it learned useful biology.

## What is single-cell RNA-seq data?

Single-cell RNA sequencing, usually called scRNA-seq, measures gene expression separately for many individual cells. Instead of averaging RNA across a tissue, it lets us compare different cell types and cell states inside the same sample.

A typical workflow looks like this:

- Collect a tissue or cell sample.
- Separate or capture individual cells.
- Add a cell barcode so RNA molecules can be assigned back to a cell.
- Sequence the RNA and map the reads to genes.
- Build a cell-by-gene count matrix.
- Perform quality control, normalization, dimensionality reduction, clustering, and cell-type annotation.

Good starting resources:

- [Scanpy preprocessing and clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering-2017.html)
- [Single-Cell Best Practices: quality control](https://www.sc-best-practices.org/preprocessing_visualizations/quality_control.html)
- [Getting started with AnnData](https://anndata.readthedocs.io/en/latest/tutorials/getting-started.html)

## What is a cell-by-gene matrix?

The main input is usually a matrix in which rows are cells and columns are genes:

|       | Gene A | Gene B | Gene C | ... |
|-------|--------|--------|--------|-----|
| Cell 1| 0      | 3      | 15     | ... |
| Cell 2| 7      | 0      | 2      | ... |
| Cell 3| 1      | 1      | 0      | ... |

Each value is usually a UMI count or another count-based measurement of how much RNA from that gene was detected in that cell. The matrix is high-dimensional and sparse: it may contain thousands or tens of thousands of genes, and many entries are zero.

A zero does not always mean that the gene is truly inactive. It may also mean that the RNA molecule was not captured or sequenced. Cells can also have very different total counts because of sequencing depth, cell size, RNA content, or cell quality.

In Python, this data is commonly stored in an AnnData object:

- `adata.X`: the main cell-by-gene matrix.
- `adata.layers["counts"]`: a common place to keep raw counts.
- `adata.obs`: cell metadata, such as batch, donor, tissue, and cell type.
- `adata.var`: gene metadata.
- `adata.obsm`: learned cell embeddings, such as PCA, UMAP, or scVI latent vectors.

For scVI and scANVI, keep the raw count matrix available. Do not give the model only a scaled or z-scored matrix.

## What is batch effect?

Batch effect means that cells look different because they were processed in different experiments, not because they are biologically different.

Batch effects can come from:

- Different laboratories, operators, reagent lots, or sequencing runs.
- Different sample preparation or tissue dissociation procedures.
- Different 10x chemistry versions or sequencing platforms.
- Different sequencing depths and capture efficiencies.
- Storage time, freezing, transport, or sample quality.
- Ambient RNA, doublets, and other technical artifacts.

The difficult part is that technical and biological factors can be mixed together. For example, if Batch A contains mostly T cells and Batch B contains mostly B cells, a model cannot perfectly decide which differences come from cell type and which come from batch without extra assumptions or labels.

This is why the goal is not simply to remove every difference between batches. Over-correction can erase real biology. A good representation should:

- Mix the same cell type across batches.
- Keep different cell types and biological states separate.
- Preserve marker genes, rare populations, trajectories, and condition-specific signals.
- Generalize to a held-out batch, donor, or study.

Useful resources:

- [Single-Cell Best Practices: data integration](https://www.sc-best-practices.org/integration.html)
- [Benchmarking atlas-level data integration in single-cell genomics](https://www.nature.com/articles/s41592-021-01336-8)
- [A benchmark of batch-effect correction methods for scRNA-seq](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02136-9)
- [Optional alternatives: Harmony and Scanorama](https://www.nature.com/articles/s41592-019-0619-0)

## How scVI handles batch effects

scVI is a variational autoencoder designed for single-cell count data. It models gene counts with a probabilistic count distribution and learns a low-dimensional latent representation for each cell.

The useful idea for our project is that scVI gives the batch label to the generative model as an observed covariate. The decoder can use both the biological latent representation and the known batch information to reconstruct the original counts. In principle, this reduces the need to store batch information in the biological latent vector.

In simple terms:

```text
cell counts -> encoder -> latent cell representation
latent representation + known batch -> decoder -> reconstructed counts
```

scVI is a strong baseline for:

- Batch-aware representation learning.
- Visualization and clustering.
- Differential expression.
- Testing whether the latent representation generalizes across datasets.

However, scVI does not guarantee that the latent space contains zero batch information. Batch leakage should still be measured with a batch classifier or integration metrics.

Practical resources:

- [Introduction to scvi-tools](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/quick_start/api_overview.html)
- [scvi-tools data preparation](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/data_loading.html)
- [Atlas-level integration tutorial using scVI and scANVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scvi_integration.html)
- [Official scvi-tools repository](https://github.com/scverse/scvi-tools)

## How scANVI is different

scANVI is a semi-supervised extension of scVI. It can use cell-type labels for some cells while also learning from unlabeled cells.

This is especially relevant to us because our downstream task is cell-type classification. The label information helps the model keep cell types separate while integrating batches. It can also transfer labels from an annotated reference dataset to a new query dataset.

The main difference is:

- scVI uses counts and batch information.
- scANVI additionally uses partial cell-type labels.

The scANVI model guide is useful because it clearly describes the model, inputs, advantages, and limitations. One important limitation listed in the documentation is that the latent space is not directly interpretable like a linear model. That limitation is close to the problem we want our VQ-VAE to address.

## What scFoundation represents

scFoundation is a large pretrained model for single-cell transcriptomics. The published model has about 100 million parameters, covers roughly 20,000 genes, and was pretrained on more than 50 million human single-cell profiles.

It represents the other side of our trade-off: strong representation learning. Its transformer-like architecture can capture complex relationships among genes and transfer to tasks such as cell-type annotation, perturbation prediction, drug-response prediction, and gene-module inference.

For our project, scFoundation is useful as:

- A strong cell-embedding baseline.
- A feature extractor or teacher model.
- A comparison for cell-type classification.
- An example of a model with strong representations but less direct interpretability.

The official code and pretrained-model information are available in the [scFoundation repository](https://github.com/biomap-research/scFoundation).

We should not assume that a foundation-model embedding is automatically batch-invariant. It should be evaluated with the same batch and biological-conservation tests as our VQ-VAE.

## Why topic models are interpretable

Topic modeling originally comes from natural-language processing. In the classic analogy:

| Text modeling           | Single-cell modeling      |
|-------------------------|---------------------------|
| Document                | Cell                      |
| Word                    | Gene                      |
| Word count              | Gene-expression count     |
| Topic                   | Gene-expression program   |
| Topic mixture           | Cell representation        |

A cell can be represented as a mixture of topics, and each topic is represented by a weighted list of genes. This makes it natural to inspect the top genes in each topic and run pathway-enrichment analysis.

The foundational paper is [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf). For single-cell RNA-seq, [scETM](https://github.com/zhanglab-aim/scETM) is particularly relevant. It combines a neural encoder with an interpretable linear decoder and explicitly models batch-specific effects. This makes scETM one of the closest existing methods to our representation-versus-interpretability question.

Additional resources:

- [A scalable approach to topic modeling in single-cell data](https://www.nature.com/articles/s41592-022-01422-5)
- [Topic modeling versus clustering of gene-expression data](https://academic.oup.com/bioinformatics/article/38/6/1625/6473259)

Topic models are not always weak, and scETM already improves their representation ability. The remaining limitation is that an additive mixture of gene programs may still miss complex nonlinear interactions, hierarchical cell states, or subtle signals needed for difficult downstream tasks.

## Where our VQ-VAE fits

The original VQ-VAE learns a discrete codebook instead of using only a continuous latent vector:

```text
cell counts -> encoder -> continuous embedding
continuous embedding -> nearest codebook vector -> discrete code
discrete code + batch information -> decoder -> reconstructed counts
```

Our working idea is to make the discrete codes both useful and biologically understandable. A good code should have several properties:

- Cells using the code share meaningful cell types, states, or biological programs.
- The genes associated with the code form coherent pathways.
- The same code has a similar meaning across batches.
- The code improves or maintains performance on cell-type classification.
- The codebook is actually used rather than collapsing to a few codes.

One possible batch-disentanglement design is:

- Use a biological codebook for cell identity and biological state.
- Give the decoder the known batch label through a separate batch embedding.
- Add a batch-classification adversary on the biological representation and reverse its gradients.
- Add a cell-type classification loss to preserve useful biological information.
- Encourage cells of the same type from different batches to have similar codes.
- Measure code usage, code stability, associated genes, and pathway enrichment.

This should be treated as a hypothesis, not as a guaranteed solution. An adversarial loss can remove useful biological information when batch and biology are strongly confounded, so it needs careful ablation and evaluation.

## The first task to work on

A good first project is to build a clean batch-effect baseline before changing the VQ-VAE.

Use one public multi-batch scRNA-seq dataset with overlapping cell types and compare:

- PCA on normalized highly variable genes.
- scVI using raw counts and a batch key.
- scANVI initialized from scVI, if cell-type labels are available.
- Our current VQ-VAE representation.

For each representation, produce:

- A UMAP colored by batch.
- The same UMAP colored by cell type.
- Cell-type classification accuracy and macro-F1.
- Cross-batch classification, where one batch or donor is held out for testing.
- Batch-classifier accuracy on the learned representation.
- Biological-conservation and batch-mixing metrics from scIB.

For the VQ-VAE, additionally report:

- Number and fraction of active codes.
- Codebook perplexity or code-use entropy.
- Code usage by cell type and by batch.
- Top genes and enriched pathways for each important code.
- Stability of code meaning across random seeds and batches.

The first goal is not to invent a new loss immediately. It is to determine where batch information enters the current model and establish a reproducible baseline that later changes can improve.

## Suggested reading order

- **Day 1: Understand the data**
  - Read the AnnData introduction.
  - Run the Scanpy preprocessing tutorial.
  - Be able to explain `X`, `obs`, `var`, `layers`, and `obsm`.

- **Day 2: Understand batch effects**
  - Read the Single-Cell Best Practices integration chapter.
  - Skim the scIB benchmarking paper.
  - List the batch variables and biological variables in the dataset you plan to use.

- **Day 3: Run scVI**
  - Read the scVI paper.
  - Run the scvi-tools quick-start tutorial.
  - Compare PCA and scVI embeddings by batch and cell type.

- **Day 4: Run scANVI**
  - Read the scANVI paper.
  - Read the scANVI model guide.
  - Compare scVI and scANVI on cross-batch cell-type classification.

- **Day 5: Connect the baselines to our model**
  - Read the scETM paper.
  - Skim the scFoundation paper.
  - Read the original VQ-VAE paper.
  - Write a short proposal for separating biological codes from batch information in our model.

## Questions to keep in mind

- What batch variables are available: study, donor, sample, platform, lane, or chemistry?
- Which biological groups are shared across batches?
- Are batch and cell type strongly confounded?
- Should the model correct batch in the latent space, the reconstructed expression matrix, or both?
- How will we prove that biological information is preserved?
- What should one discrete VQ code mean biologically?
- Can the same code be linked to similar genes and pathways in different batches?
- Does improved interpretability reduce downstream accuracy, and how much reduction is acceptable?
