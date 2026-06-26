# Benchmark pattern comparison on datasets with curated positive controls

This folder builds the comparator-benchmark figures for the QuadSV paper.
The analysis compares cross-sample differential frequency (DF) tests,
implemented by the `Comparator` class, against a pseudo-bulk differential
expression (DE) baseline for both AD/PIG and brain-development benchmarks.

- `workflow.py` runs analysis and caches results under
  `results/comparator_benchmark/`.
- `visualization.ipynb` generates the manuscript-style main and supplementary
  figures under `figures/comparator_benchmark/`.

## Minimal Dependencies

Install the non-`quadsv` dependencies from the root environment YAML file.
Apply any additional `env*.yml` overlays the same way.

```bash
mamba env create -f environment.yml
```

For an existing environment:

```bash
mamba env update -n quadsv_paper -f environment.yml --prune
```

Install `quadsv` interactively from the local source checkout with its dev and
docs optional dependencies. By default `config.py` records the source path as
`../quadsv`; override `QUADSV_SRC` if needed.

```bash
QUADSV_SRC="${QUADSV_SRC:-$(cd .. && pwd)/quadsv}"
conda run -n quadsv_paper python -m pip install --no-build-isolation -e "${QUADSV_SRC}[dev,docs]"
```

On this machine, full comparator runs are most stable with the thread limits
shown below. Project config sets `NUMBA_CACHE_DIR` to `.cache/numba` before
Scanpy is imported, which avoids Scanpy/Numba cache-locator failures while
keeping JIT enabled.

## How To Run

All commands assume the repository root is the current working directory.

Define the common runtime wrapper once per shell:

```bash
mkdir -p /private/tmp/mplconfig /private/tmp/xdg-cache

run_quadsv_paper() {
  env \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    FINUFFT_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/private/tmp/mplconfig \
    XDG_CACHE_HOME=/private/tmp/xdg-cache \
    conda run -n quadsv_paper "$@"
}
```

Prepare the local data cache when needed:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.download_ad_pig_data

run_quadsv_paper python -m scripts.comparator_benchmark.download_brain_dev_data
```

Run the cached AD/PIG analysis workflow:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.workflow --analysis ad-pig
```

The AD/PIG workflow runs three high-level analysis components used by
`ad_main.pdf` and several `ad_supp_*.pdf` figures:

1. Full requested-30-bin scenario grid: raw/log1p input, linear/log binning, radial/2D
   features, observed DE/DF/DF-norm tests, and DF/DF-norm permutation ECDFs.
2. Raw radial bin-count robustness: effective ranks, permutation FPR, FDR-hit
   counts, reference p-value correlations, and reference-hit recovery.
3. Reference analysis extras for raw/linear/radial/30 bins: top-hit
   clusterProfiler enrichment, DF-vs-DF-norm enrichment contrasts, raw 2D
   spectrum anisotropy, and technical-replicate spectral diagnostics.

Run the continuous-design brain-development workflow:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.workflow --analysis brain-dev
```

It uses anterior-plane Visium samples, `log10(PN day + 1)` as the continuous
design, and the local Cell 2026 Table S4 supplement under
`data/brain_dev/supplementary/table_s4_cnmf_programs.xlsx`. It caches:

1. A four-scenario grid: raw/log1p input crossed with radial/2D features,
   linear-bin frequencies, observed DE/DF/DF-norm tests, and 30
   between-timepoint permutation ECDFs.
2. Reference raw/linear-bin/2D extras: cell-type/process positive controls,
   marker/background anisotropy, and top-100 DF-vs-DF-norm enrichment contrasts.

Run all currently implemented workflows. This is the default when `--analysis`
is omitted:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.workflow
```

Use `--force` to recompute cached outputs for the selected analysis:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.workflow --analysis ad-pig --force
```

Use `--max-genes` or `--presence-threshold` for smaller local runs. For a fast
smoke test without overwriting existing caches, use a small gene cap:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.workflow --analysis brain-dev --max-genes 25
```

Render the manuscript figures from the notebook after the relevant workflow
mode has completed:

```bash
run_quadsv_paper python -m jupyter nbconvert --to notebook --execute \
  scripts/comparator_benchmark/visualization.ipynb \
  --output /private/tmp/comparator_benchmark_visualization.executed.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=quadsv_paper
```

The notebook writes PDF figures under `figures/comparator_benchmark/`.

The standalone AD/PIG gene-profile plotter is intentionally argument-driven:
it does not choose a gene panel or output filename. Use it for custom follow-up
profiles after the workflow caches exist:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.plot_ad_pig_gene_profiles \
  --genes Apoe Tyrobp \
  --output figures/comparator_benchmark/ad_gene_profile_custom.pdf
```

The brain-development profile plotter follows the same pattern:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.plot_brain_dev_gene_profiles \
  --genes Fabp7 Synpr \
  --output figures/comparator_benchmark/brain_dev_gene_profile_custom.pdf
```

## Cached Outputs

`workflow.py` writes:

(For the AD/PIG analysis)

- `ad_pig_full_tests.csv` - observed DE, DF, and DF-norm p-values for the
  eight AD/PIG scenarios: raw/log1p input, linear/log binning, and radial/2D
  features at the requested 30 radial bins. For log binning, the workflow
  starts from a pure geometric grid and collapses empty intervals; therefore
  `requested_n_radial_bins` records the configured target and `n_radial_bins`
  records the actual occupied bin count used by the comparator.
- `ad_pig_full_null_ecdf.csv` - permutation-null DF and DF-norm p-values for
  those same eight scenarios. This cache intentionally stores only p-values
  and scenario metadata because ECDF plots do not need per-gene fields.
- `ad_pig_radial_bin_robustness.csv` - raw/radial-only bin-count diagnostics:
  per-sample raw spectrum effective-rank vectors, learned residual covariance
  effective rank, permutation FPR, FDR-hit counts, p-value Spearman correlation
  to the same-binning requested-30-bin reference, and reference-hit recovery.
  As above, `requested_n_radial_bins` and `n_radial_bins` distinguish
  configured and actual occupied bin counts for collapsed log grids. Per-sample
  rank vectors are stored as paired JSON columns,
  `sample_effective_rank_samples` and `sample_effective_rank_values`.
- `ad_pig_anisotropy.csv` - raw 2D spectrum anisotropy scores for PIG and
  non-PIG background genes. Sample summaries are derived in the visualization
  notebook.
- `ad_pig_spectral_replicate_distance.csv` - long-form same-condition
  section-pair distances for AD/PIG technical-replicate diagnostics. Each row
  is one `gene x bin x comparison`; `bin=DC` stores the explicit DC component
  and `bin=k01..` stores non-DC radial bins from the reference raw/radial
  spectrum.
- `ad_pig_enrichment_contrasts.csv` - AD/PIG top-100 DF versus DF-norm
  enrichment contrast table used by the main figure and AD/PIG gene-profile
  selection.

(For the brain-development analysis)

- `brain_dev_tests.csv` - observed DE, DF, and DF-norm p-values for the
  continuous postnatal-time brain-dev four-scenario grid: raw/log1p input,
  radial/2D features, linear-bin frequencies, and requested 30 radial bins.
  The reference scenario for main figures is raw/linear-bin/2D.
- `brain_dev_null_ecdf.csv` - permutation-null DF and DF-norm p-values for
  brain-dev ECDF calibration. The workflow keeps only p-values and scenario
  metadata from 30 between-timepoint label permutations per scenario/test.
- `brain_dev_anisotropy.csv` - raw 2D spectrum anisotropy scores for
  brain-dev marker and non-marker background genes.
- `brain_dev_positive_controls.csv` - paper-derived positive-control genes
  from Cell 2026 Table S4 cNMF programs plus manual curation, grouped by cell
  type or process signatures such as immature neurons, mature synaptic neurons,
  astrocytes, oligodendrocytes, and vascular/mural cells. Manual rows use
  `Cell 2026 manual curation` as their source.
- `brain_dev_enrichment_contrasts.csv` - brain-dev top-100 DF versus DF-norm
  clusterProfiler GO BP / KEGG enrichment contrast table.

For both analyses, we also cache the full enrichment responses under 
`enrichment_cache/*.csv`:

- cached clusterProfiler GO BP / KEGG enrichment responses and top-gene lists.
  Enrichment tables retain per-term hit genes in `hit_genes`. 
- Raw clusterProfiler term names are kept in `name`; sentence-style
  labels for plotting are stored separately in `display_name`. 
- GO BP terms also include `semantic_simplified`, marking terms retained by
  `clusterProfiler::simplify(cutoff=0.7, measure="Wang")`; all original GO terms
  remain in the cache.

No broader benchmark sweep table is written because it is not needed for the
current figure outputs.

For these cached figure inputs, the workflow runs:

- Spatial pattern test: `log_l2` with QuadSV's analytic null on
  `cmp.spectra_`.
- DE companion: Welch t for binary panels or QuadSV scalar GLM for the
  continuous panel on `cmp.dc_`.

The full-test table stores observed DE, DF, and DF-norm p-values. The
permutation ECDF table stores DF and DF-norm p-values under label shuffling.
FDR adjustments for figure summaries stay in the notebook unless the cache
explicitly stores FDR-level diagnostics.

Compute/cache split:

- Keep in `workflow.py`: fitted-comparator p-value tables, permutation-null
  ECDF p-value pools, bin-count robustness diagnostics, enrichment responses,
  enrichment contrasts, anisotropy, and AD/PIG replicate spectra. These are
  expensive, deterministic analysis products and are reused across figures.
- Keep in `visualization.ipynb`: figure assembly, schematic drawing, spatial
  maps, PSD curves, and panel-specific layout choices. These are plot-specific
  and easier to inspect interactively.
- Keep in `plot_common.py`: small plotting/data helpers shared by the notebook
  and standalone plotters, such as paper Matplotlib defaults, ECDF values,
  p-value transforms, scenario filtering, and pathway-label formatting.

## Data Preparation

The benchmark expects per-sample `.h5ad` files and metadata CSVs under
`data/ad_pig/` and `data/brain_dev/`. These idempotent scripts prepare
the local data cache:

```bash
run_quadsv_paper python -m scripts.comparator_benchmark.download_ad_pig_data

run_quadsv_paper python -m scripts.comparator_benchmark.download_brain_dev_data
```

AD/PIG probe QC is applied when `load_panel("ad_pig")` reads those `.h5ad`
files, before gene selection or comparator fitting. Probes with design issues
are removed (`H2-D1`, `Cd63-ps`, `RP23-269H21.1`), and cross-reactive probes
are renamed/collapsed to shared family symbols (`C4a`/`C4b` -> `C4`,
`Serpina3n` -> `Serpina3`, `Lyz2` -> `Lyz`). The raw downloaded files are left
unchanged.

Brain-dev QC is applied during `.h5ad` generation by
`download_brain_dev_data.py`: unique variable names, in-tissue spots only,
genes present in at least 3 spots, spots with 3,000-50,000 total counts, and a
final gene-presence filter. The downloader also stores Visium coordinates in
micrometers using the 55 um capture-spot diameter, so `load_panel("brain_dev")`
does not rescale coordinates.

## Layout

```text
scripts/comparator_benchmark/
├── README.md
├── __init__.py
├── ad_pig_helpers.py
├── brain_dev_helpers.py
├── config.py
├── download_ad_pig_data.py
├── download_brain_dev_data.py
├── plot_ad_pig_gene_profiles.py
├── plot_brain_dev_gene_profiles.py
├── plot_common.py
├── utils.py
├── workflow.py
└── visualization.ipynb
```

## Code Checks

```bash
conda run -n quadsv_paper python -m compileall -q scripts/comparator_benchmark
conda run -n quadsv_paper python -m scripts.comparator_benchmark.workflow --help
```
