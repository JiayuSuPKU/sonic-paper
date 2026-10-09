# Human glioma IDH-status comparison

This folder builds the IDH-mutant vs
IDH-wildtype glioma Visium expression-pattern comparison figures for the SONIC paper.

- `download_glioma_data.py` downloads and prepares the Greenwald et al. Visium data using the same QC specifications as the original paper.
- `workflow.py` runs the IDH-wildtype vs IDH-mutant DE/DF tests and caches figure inputs under `results/glioma_idh/`.
- `visualization.ipynb` previews each panel and renders the combined manuscript-style main and supplementary figures under
  `figures/glioma_idh/`.

## How to Run:

Run from the project root with the `sonic_paper` environment, available from [`environment.yml`](../../environment.yml):

```bash
conda run -n sonic_paper \
  python -m scripts.glioma_idh.workflow
```

Add `--force` to recompute all stages. Otherwise, current stages are reused.

The downloader, workflow, and figures all use `DATA_ROOT` from `config.py`.
The configured root and input files are tracked in cache provenance, so changing them invalidates old results.
Prepared inputs must match the current QC settings and micrometer-coordinate cache version.
Rerun the downloader if those checks fail.

The glioma analysis workflow runs five main steps to generate inputs for the `idh_main.pdf`, `idh_supp.pdf`, and per-program `idh_<metaprogram>.pdf` figures:

1. IDH-status DE (differential expression) and DF/DF-norm tests on the top 7,000 genes by mean log-normalized expression after spot QC.
2. Pathway enrichment of the top 100 DE and top 100 DF-norm genes with clusterProfiler.
3. Meta-program mapping for pathways (significant, DF-norm-preferred) and top 100 DF-norm genes using keyword rules in `config.py`. Genes supported by pathways from multiple programs are assigned to the highest-scoring program, while also retaining all pathway matches for inspection.
4. Permutation null calibration of DF and DF-norm with 50 sample-label shuffles.
5. Additional diagnostic properties such as per-gene anisotropy and per-sample effective rank of the radial spectra.

## Literature Context for Retained Meta-programs

The five retained meta-programs are literature-guided summaries of the
enrichment results, not independent pathway tests. DF-norm establishes that
spatial pattern shape differs by IDH status; it does not by itself determine
the direction of pathway activity or prove direct IDH causality. Grade,
histology, necrosis, lineage, and tumor-microenvironment composition remain
important potential confounders.

- **Metabolism / redox:** IDH mutations produce 2-hydroxyglutarate and alter
  amino-acid, energy, NAD+, and oxidative-stress metabolism. Relevant studies
  include [Dang et al., *Nature* (2009)](https://www.nature.com/articles/nature08617),
  [Tonjes et al., *Nature Medicine* (2013)](https://www.nature.com/articles/nm.3217),
  [Tateishi et al., *Cancer Cell* (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4684594/),
  and an [IDH-mutant glioma metabolic profiling study (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5709746/).
- **Hypoxia / vascular:** IDH status has been associated with distinct
  hypoxia, angiogenesis, and vascular transcriptional programs
  ([Gilbert et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4633672/);
  [lower-grade glioma vascular analysis (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6176806/)).
- **Immune / complement:** Mutant IDH and 2-hydroxyglutarate influence immune
  recruitment and immunosuppressive myeloid and T-cell states
  ([Amankulor et al., *Genes & Development* (2017)](https://pubmed.ncbi.nlm.nih.gov/28465358/);
  [Kohanbash et al., *JCI* (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5373859/);
  [Friedrich et al., *Nature Cancer* (2021)](https://www.nature.com/articles/s43018-021-00201-z)).
- **Plasminogen / coagulation:** The association between IDH status and
  thrombosis is unusually direct, including methylation-dependent suppression
  of tissue factor in IDH-mutant glioma
  ([Unruh et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5640980/);
  [tissue-factor study (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6537089/)).
- **ECM / integrin:** IDH-linked epigenetic state and vascular context can
  affect extracellular-matrix, adhesion, and migration programs
  ([glioma migration and ECM-adhesion study (2024)](https://pubmed.ncbi.nlm.nih.gov/39596246/);
  [Tumor-TAM interaction (2024)](https://link.springer.com/article/10.1186/s40478-024-01837-5);
  [ECM-signature (2025)](https://www.nature.com/articles/s41698-025-01100-7)).

## Cached Outputs:

The workflow writes:

- `idh_status_tests.csv` - DE, DF, and DF-norm p-values/FDRs for the selected gene panel.
- `idh_status_permutation_null.csv` - DF and DF-norm p-values/FDRs from 50 deterministic sample-label shuffles, retaining only fields needed for null calibration.
- `idh_status_spectra.npz` - generated cache of radial spectra, per-sample gene-presence masks, anisotropy, sample IDs, groups, h5ad paths, and genes used by `visualization.ipynb`.
- `idh_status_spatial_diagnostics.csv` - per-gene/sample raw 2D anisotropy plus per-sample expression and radial-power effective ranks.
- `idh_status_clusterprofiler_enrichment.csv` - merged GO BP and KEGG clusterProfiler results for the top 100 DE and DF-norm genes.
- `idh_status_df_norm_top100_metaprogram_map.csv` - deterministic pathway-based assignments of the top 100 DF-norm genes to the configured meta-programs.
- `cache_manifest.json` - input signatures and output fingerprints that prevent reuse of incompatible comparison or enrichment caches; the enrichment entry also records the resolved R command, library paths, wrapper checksum, and core package versions.

Comparison, diagnostics, enrichment, and meta-program mapping have separate
cache entries. Missing diagnostics can be rebuilt from saved spectra and
normalized expression. Plotting edits do not invalidate numerical stages.
R-wrapper edits invalidate enrichment only (and its downstream map).
Successful enrichment jobs with no terms are cached as empty tables.

## Rendering Figures:

After a successful workflow run, render the figure from cached test, enrichment,
spectra, diagnostics, and bundled sample-path metadata with:

```bash
conda run -n sonic_paper python -m jupyter nbconvert --to notebook --execute \
  scripts/glioma_idh/visualization.ipynb \
  --output /private/tmp/glioma_idh_visualization.executed.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=sonic_paper
```

The notebook includes per-panel previews before each save cell. It derives four
single-page `idh_<metaprogram>.pdf` figures from the
non-metabolic programs in `config.py`, excluding genes already shown in the
main figure. The standalone
`plot_gene_profiles.py` CLI remains available for ad hoc gene selections.

Each gene panel shows raw radial profiles beside the per-bin log2 fold change
of L1-normalized group means. Use `--sample-ids IDHm_BWH23O GBM_ZH8811Bbulk`
to fix the spatial maps shown in each gene panel instead of using per-gene
representative samples. Profiles and automatic representative-sample selection use the
same per-gene presence mask as DF testing.

The notebook and standalone profile loader validate cache provenance before
reading results. Older caches need one workflow run to regenerate the new
metadata.