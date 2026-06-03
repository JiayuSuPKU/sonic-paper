# Atera FFPE Human BRCA ANKRD30A Analysis

This folder contains the cache-first workflow and visualization notebook for the
Atera FFPE human BRCA SV/DU analysis. The analysis uses segmented
`table_codeword` data, `SplisosmNP`, and the ANKRD30A event as the focused
long-read/TCGA validation example.

SV is run on all segmented codeword features with `hsic-ir`. DU is run once for
all SVP genes (`SV pvalue_adj < 0.01`) against one combined regulator design
matrix containing configured POSTAR3 RBPs and ReMap human TF-ChIP covariates
(`hsic-gp`, `gpr_backend="nufft"`). The canonical filters are
`min_counts=100`, `min_bin_pct=0.05`, `min_component_size=100`,
`rho=0.99`, and `k_neighbors=4`.

## How To Run

All commands assume the repository root is the current working directory.

```bash
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/workflow.py --n-jobs 8
```

Use `--force` to recompute cached outputs:

```bash
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/workflow.py --force --n-jobs 8
```

By default, `workflow.py` runs SV, all-SVP DU, all-SVP probe genome/transcript
mapping, top-20 SVP PNG inspection panels, and the ANKRD30A long-read/TCGA
summary analysis, including the T47D TRPS1-degradation PRO-seq cache. 
It does not render the final PDFs. Use `--skip-top20` or
`--skip-ankrd30a` to omit optional downstream stages.

Render final figures manually from `visualization.ipynb`. The notebook owns the
figure-specific panel code and reads only workflow-generated caches.

Standalone downstream entry points:

```bash
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/plot_top20_svp_probe_transcript_panels.py --render
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/run_ankrd30a_analysis.py
```

The standalone `run_ankrd30a_analysis.py` file wraps the ANKRD30A-specific analysis, which lives in the `ankrd30a/` subfolder:

- `atera_du.py`: Atera ANKRD30A DU ranking and ReMap PB-TSS binding support.
- `long_read.py`: PacBio annotation, ER+ FL-count summaries, and notebook caches.
- `download_tcga_brca_ankrd30a.py` and `download_tcga_brca_clinical_expression.py`: TCGA source-data acquisition.
- `tcga.py`: TCGA BAM quantification, ER status/metric assembly, and ER+ regulator correlations.
- `run_trps1_proseq.py`: GSE236172 ANKRD30A-region PRO-seq download and PB TSS/gene-body signal caches.
  Use `--dry-run` to test the workflow and `--force` to re-download cached files.

## Analysis Policy

- SV uses segmented `table_codeword`, `SplisosmNP`, and `hsic-ir`.
- DU targets are all canonical SVP genes with `SV pvalue_adj < 0.01`.
- DU uses one combined regulator matrix from configured POSTAR3 RBPs plus ReMap
  2022 human hg38 TF-ChIP covariates.
- DU uses `hsic-gp` with `gpr_backend="nufft"` and no spatial kernel in the
  regulator test stage.
- Feature/regulator filtering uses `min_counts=100` and `min_bin_pct=0.05`;
  spot filtering uses `min_component_size=100`.
- ANKRD30A interpretation focuses on PacBio QC-pass transcript classes
  compatible with `45213+50762` versus `50762+7939`.
- TCGA validation quantifies only PacBio QC-pass ANKRD30A transcripts with
  StringTie/prepDE and counts Atera probe regions directly with featureCounts.
  Probe-region TCGA counts do not use probe-to-transcript assignment.
- TRPS1-KO PRO-seq validation uses only ANKRD30A-region slices of GSE236172
  processed bigWigs, summarizes TSS signal (total normalized signal in 
  +/-1Kb TSS region, both strands) and compares to control (0h) across 
  TRPS1 degradation timepoints.

## Inputs

Local analysis inputs live under:

```text
data/atera_ffpe_human_brca/
```

Reference inputs live under `~/reference/`:

- `hg38/gencode.v49.annotation.sorted.gtf.gz`
- `hg38/gencode.v49.annotation.sorted.gtf.gz.tbi`
- `ReMap2022/remap2022_nr_macs2_hg38_v1_0.bed.gz`
- `ReMap2022/remap2022_tf_targets_hg38.csv`

Download helpers are intentionally small and idempotent:

```bash
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/download_pb_brca.py
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/ankrd30a/download_tcga_brca_ankrd30a.py --token-file /path/to/gdc-token.txt --threads 8
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python scripts/atera_ffpe_human_brca/ankrd30a/download_tcga_brca_clinical_expression.py
```

Runtime settings, tool paths, regulator lists, and output folders live in
`config.py`.

## Outputs

Results are written under:

```text
results/atera_ffpe_human_brca/
```

Main cached files:

- `segmented_splisosmnp_svp_hsic_ir.csv`
- `du_regulator_segmented_svp_hsic_gp_nufft.csv`
- `du_regulator_status.csv`
- `summary.json`
- `svp_probe_transcript_cache/svp_probe_mappings.csv`
- `svp_probe_transcript_cache/svp_gencode_transcript_probe_compatibility.csv`
- `svp_probe_transcript_cache/svp_pacbio_transcript_probe_compatibility.csv`
- `ankrd30a_visualization_cache/ankrd30a_transcript_annotations.csv`
- `ankrd30a_visualization_cache/ankrd30a_probe_mappings.csv`
- `ankrd30a_visualization_cache/ankrd30a_regulator_priority.csv`
- `tcga_brca_ankrd30a/tcga_brca_ankrd30a_pb_probe_metric_table.tsv`
- `tcga_er_positive_regulator_correlations.csv`
- `ankrd30a_trps1_proseq/gse236172_sample_manifest.csv`
- `ankrd30a_trps1_proseq/ankrd30a_proseq_signal_windows.csv`
- `ankrd30a_trps1_proseq/ankrd30a_proseq_signal_changes.csv`

The workflow treats `results/atera_ffpe_human_brca/` as a generated cache.
The all-SVP probe mapping cache includes metadata that records the requested SVP
gene set, source-file fingerprints, and mapping parameters before cache reuse.

Figures are written by `visualization.ipynb` under:

```text
figures/atera_ffpe_human_brca/
```

Publishable PDFs:

- `atera_ankrd30a.pdf`
- `atera_ankrd30a_du_tcga.pdf`

Top-20 SVP inspection PNGs are kept separate under:

```text
figures/atera_ffpe_human_brca/top20_svp_probe_transcript_panels/
```

## Code Checks

```bash
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python -m ruff check scripts/atera_ffpe_human_brca/*.py scripts/atera_ffpe_human_brca/ankrd30a
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python -m black --check scripts/atera_ffpe_human_brca/*.py scripts/atera_ffpe_human_brca/ankrd30a
/Users/jysumac/miniforge3/envs/quadsv_dev/bin/python -m compileall -q scripts/atera_ffpe_human_brca
```

The notebook is intentionally excluded from ruff/black because it is the manual
figure-rendering surface.
