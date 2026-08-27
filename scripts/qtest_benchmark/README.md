# Q-test benchmarks on synthetic and real data

This folder validates Q-test implementations, benchmarks runtime and memory,
illustrates kernel-dependent pattern prioritization in Visium HD mouse small
intestine, and tests graph-dependent patterns in KP-Tracer lineage trees.

- `workflow.py` builds every analysis cache required by the figures, including the following modules:
  - `benchmark.py` owns runtime scaling, implementation agreement, null
  calibration, and power simulations.
  - `small_intestine.py` computes transcriptome-wide CAR sensitivity and
  contrastive spectral-kernel results directly from SpatialData.
  - `lineage.py` computes cluster-, gene-, and pair-level Moran/CAR results and
  selected spectral diagnostics from the lineage trees.
- `visualization.ipynb` previews each panel and renders three main and two
  supplementary figures.
- `config.py` defines all input, cache, figure, cohort, and analysis settings.

## How to Run

Run from the project root with the `quadsv_paper` environment from
[`environment.yml`](../../environment.yml). Thread limits keep the reported
runtime benchmark single-core:

```bash
conda run -n quadsv_paper \
  python -m scripts.qtest_benchmark.workflow
```

A normal run validates each cache and skips current work. `--force` recomputes
derived caches. Use `--analysis synthetic`, `--analysis small-intestine`, or
`--analysis lineage` to run one cache group; the default is `all`.

The workflow expects two repository-local inputs:

- `data/qtest_benchmark/visium_hd_mouse_intestine.zarr`: the 10x Genomics
  Visium HD mouse small-intestine SpatialData store. `--source` can override
  this path without changing the configuration.
- `data/qtest_benchmark/lineage_tracing/`: the KP-Tracer expression subset and
  per-tumor Newick trees.

The small-intestine source is available from the
[10x dataset page](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine)
and the
[SpatialData archive](https://s3.embl.de/spatialdata/spatialdata-sandbox/visium_hd_3.0.0_io.zip). 
The lineage-tracing source is available from the
[KP-Tracer Zenodo repository](https://zenodo.org/records/5847462#.YrFDKuzMI6A).

The workflow runs four stages in dependency order:

1. Benchmark build-plus-test runtime and memory up to each implementation's
   declared sample limit, then project transcriptome-wide costs under an 8 GB
   memory budget.
2. Cache seeded implementation-agreement, null-calibration, and power results
   on regular and irregular grids.
3. Load the 16 µm small-intestine table, compute transcriptome-wide CAR scores
   across the configured autocorrelation sweep, and fit the two contrastive
   pattern spectra.
4. Build lineage-graph operators from Newick trees and cache cross-tumor
   cluster, gene, bivariate, and selected spectral diagnostics. Cluster tests
   include every eligible tree; the much costlier gene and pair tests are
   skipped, with a log message, for trees above 3,000 cells.

The analysis modules expose cache functions for targeted interactive use;
`workflow.py` is the single command-line entry point.

## Cached Outputs

- `results/qtest_benchmark/benchmark_{scaling,projection}.csv`: measured
  single-call costs and transcriptome-wide projections.
- `results/qtest_benchmark/validation/`: implementation agreement, Welch/Liu
  comparison, false-positive-rate, power, and hyperparameter-sweep tables.
- `results/qtest_benchmark/small_intestine_{scores,pattern_ranks}.csv`:
  transcriptome-wide CAR scores, group ranks, and optimized spectral-fit
  summaries.
- `results/qtest_benchmark/lineage_tracing/`: six plot-ready cluster, gene,
  bivariate, spectral-diagnostic, and target-metadata caches.

The notebook loads the eight displayed small-intestine genes directly from the
SpatialData zarr; it does not maintain gene-specific expression caches. Cheap
lineage summaries and tree layouts are reconstructed in memory.

## Rendering Figures

After the workflow completes, execute the notebook:

```bash
conda run -n quadsv_paper python -m jupyter nbconvert --to notebook --execute \
  scripts/qtest_benchmark/visualization.ipynb \
  --output /private/tmp/qtest_benchmark_visualization.executed.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=quadsv_paper
```

The notebook previews every panel before saving flat under
`figures/qtest_benchmark/`. Previews remain in the executed notebook; only PDFs
are written to the figure directory:

- Main: `main_framework.pdf`.
- Main: `main_runtime.pdf`.
- Supplementary: `supp_validation.pdf`.
- Main: `main_lineage.pdf`.
- Supplementary: `supp_lineage.pdf`.
