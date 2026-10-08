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

Run from the project root with the `sonic_paper` environment from
[`environment.yml`](../../environment.yml). Explicit thread limits keep
single-core scaling measurements separate from the multicore projection:

```bash
conda run -n sonic_paper \
  python -m scripts.qtest_benchmark.workflow
```

A normal run validates each cache and skips current work. `--force` recomputes
derived caches. Use `--analysis runtime`, `--analysis validation`,
`--analysis small-intestine`, or `--analysis lineage` to run one cache group;
the default is `all`. Runtime and validation use generated inputs and run independently.

The small-intestine and lineage analyses expect two repository-local inputs:

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

With `--analysis all`, the workflow runs these four independent stages:

1. Benchmark build-plus-test runtime and memory up to each implementation's
   declared sample limit on one core, then measure multicore throughput at each
   target size and project 10,000-feature compute time with up to 8 cores and
   16 GiB total monitored process memory.
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
`workflow.py` is the only command-line entry point.

## Cached Outputs

- `results/qtest_benchmark/benchmark_{scaling,projection}.csv`: single-core
  scaling and target-specific multicore projections, including phase timings,
  dispersion, resolved concurrency/chunks, calibration, and failure status.
- `results/qtest_benchmark/benchmark_runs.jsonl`: every independent measurement
  repetition, including failed or interrupted-by-budget cases.
- `results/qtest_benchmark/validation/`: implementation agreement, Welch/moment-calibration
  comparison, false-positive-rate, power, and hyperparameter-sweep tables.
- `results/qtest_benchmark/small_intestine_{scores,pattern_ranks}.csv`:
  transcriptome-wide CAR scores, group ranks, and optimized spectral-fit
  summaries.
- `results/qtest_benchmark/lineage_tracing/`: cluster, gene,
  bivariate, spectral-diagnostic, and target-metadata caches, plus `qtest_qc.csv`
  recording input/retained cells, support cutoffs, eligible features, and
  discoveries for each tumor, feature type, and kernel.

The notebook loads the eight displayed small-intestine genes directly from the
SpatialData zarr's `tables/square_016um` AnnData store.
Cheap lineage summaries and tree layouts are reconstructed in memory.

## Runtime measurement protocol

- **Single-core scaling:** kernel construction, default null calibration, and
  one Q-test. Three fresh-process repetitions; report median time and maximum RSS.
- **10,000-gene projection:** setup + 10,000 / measured throughput at each target
  size, using SONIC's automatic scheduling with up to **8 cores and 16 GiB total
  RSS**. Streamed batches share the remaining workspace after retained memory,
  input buffers, and 2 GiB headroom; this differs from SONIC's workspace-only budget.
- **Scope:** kernel/Q-test compute only. Excludes input generation, loading,
  preprocessing, and result assembly. Matrix measurements use shared-memory
  threads; the public matrix detector uses processes.
- **Geometry:** irregular runtime targets use unit grids jittered by ±0.25 per
  axis (seed 42). NUFFT infers its grid, capped at 1024 per axis, so resolution
  can vary with size. Features are dense Gaussian noise.
- **Nulls:** Welch for matrix Matérn/CAR, CLT for Moran, and moments for FFT/NUFFT
  Matérn; NUFFT includes 60 higher-trace probes. Failed or capped cases are not
  extrapolated into successful projections.

## Rendering Figures

The notebook validates source, input, and output fingerprints before loading
each cache group. It rejects legacy or stale caches and uses the small-intestine
source recorded by the scoring run, including a custom `--source` path.

After the workflow completes, execute the notebook:

```bash
conda run -n sonic_paper python -m jupyter nbconvert --to notebook --execute \
  scripts/qtest_benchmark/visualization.ipynb \
  --output /private/tmp/qtest_benchmark_visualization.executed.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  --ExecutePreprocessor.kernel_name=sonic_paper
```

The notebook previews every panel before saving flat under
`figures/qtest_benchmark/`. Previews remain in the executed notebook; only PDFs
are written to the figure directory:

- Main: `main_framework.pdf`.
- Main: `main_runtime.pdf`.
- Supplementary: `supp_validation.pdf`.
- Main: `main_lineage.pdf`.
- Supplementary: `supp_lineage.pdf`.
