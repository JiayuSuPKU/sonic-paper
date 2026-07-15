# Human glioma IDH-status comparison

This folder builds the IDH-mutant vs
IDH-wildtype glioma Visium expression-pattern comparison figures for the QuadSV paper.

- `download_glioma_data.py` downloads and prepares the Greenwald et al. Visium data using the same QC specifications as the original paper.
- `workflow.py` runs the IDH-wildtype vs IDH-mutant DE/DF tests and caches figure inputs under `results/visium_glioma_idh_status/`.
- `visualization.ipynb` previews each panel and renders the combined manuscript-style main and supplementary figures under
  `figures/visium_glioma_idh_status/`.

## How to Run:

Run from the project root with the `quadsv_paper` environment, available from [`environment.yml`](../../environment.yml):

```bash
conda run -n quadsv_paper \
  python -m scripts.visium_human_glioma.workflow --force
```

If the `.h5ad` files live outside this paper repo, pass the existing cache root:

```bash
conda run -n quadsv_paper \
  python -m scripts.visium_human_glioma.workflow \
  --data-root /path/to/visium_glioma_idh_status --force
```

The glioma analysis workflow runs five main steps to generate inputs for the `idh_status_expression_patterns.pdf` and `idh_status_supp_diagnostics.pdf` figures:

1. IDH-status DE (differential expression) and DF/DF-norm tests on the top 7,000 genes by mean log-normalized expression after spot QC.
2. Pathway enrichment of the top 100 DE and top 100 DF-norm genes with clusterProfiler.
3. Meta-program mapping for pathways (significant, DF-norm-preferred) and top 100 DF-norm genes using keyword rules in `config.py`. Genes supported by pathways from multiple programs are assigned to the highest-scoring program, while also retaining all pathway matches for inspection.
4. Permutation null calibration of the three tests with 20 sample-label shuffles.
5. Additional diagnostic properties such as per-gene anisotropy and per-sample effective rank of the radial spectra.

## Cached Outputs:

The workflow writes:

- `idh_status_tests.csv` - DE, DF, and DF-norm p-values/FDRs for the selected gene panel.
- `idh_status_permutation_null.csv` - DF and DF-norm p-values/FDRs from 20 deterministic sample-label shuffles, retaining only fields needed for null calibration.
- `idh_status_spectra.npz` - generated cache of radial spectra plus sample IDs, groups, h5ad paths, and genes used by `visualization.ipynb`.
- `idh_status_spatial_diagnostics.csv` - per-gene/sample raw 2D anisotropy plus per-sample expression and radial-power effective ranks.
- `idh_status_clusterprofiler_enrichment.csv` - merged GO BP and KEGG clusterProfiler results for the top 100 DE and DF-norm genes.
- `idh_status_df_norm_top100_metaprogram_map.csv` - deterministic pathway-based assignments of the top 100 DF-norm genes to the configured meta-programs.


## Rendering Figures:

After a successful workflow run, render the figure from cached test, enrichment,
spectra, diagnostics, and bundled sample-path metadata with:

```bash
conda run -n quadsv_paper python -m jupyter nbconvert --to notebook --execute \
  scripts/visium_human_glioma/visualization.ipynb \
  --output /private/tmp/visium_glioma_idh_status_visualization.executed.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=quadsv_paper
```

The notebook includes per-panel preview cells before the final combined PDF
save cell.

Generate a standalone multi-gene spatial/profile PDF from the same caches with:

```bash
conda run -n quadsv_paper python -m scripts.visium_human_glioma.plot_gene_profiles \
  --genes \
    VEGFA HILPDA SLC2A1 DCN \
    C1QA C1QB LGALS3 C1QC \
    ANXA2 ENO1 PGK1 A2M \
    COL1A1 COL6A1 COL1A2 TNC \
  --genes-per-page 4 \
  --page-titles \
    "Hypoxia / vascular organization" \
    "Immune / myeloid infiltration" \
    "Plasminogen activation / coagulation / VTE" \
    "ECM / integrin organization" \
  --output figures/visium_glioma_idh_status/idh_status_gene_profiles.pdf
```

Each page corresponds to one literature-prioritized discovery program and
shows four representative top-100 DF-norm hits. IDH-linked metabolism controls
are shown in `idh_status_supp_diagnostics.pdf`; the standalone profile PDF
retains the four other programs and excludes the main-figure genes.

Each gene panel shows raw radial profiles beside the per-bin log2 fold change
of L1-normalized group means. Use `--sample-ids IDHm_BWH23O GBM_ZH8811Bbulk`
to fix the spatial maps shown in each gene panel instead of using per-gene
representative samples.
