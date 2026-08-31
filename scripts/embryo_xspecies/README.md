# Embryo radial-spectral embedding

This folder compares per-`(gene, section)` spatial patterns across MOSTA mouse
and HESTA human embryo Stereo-seq sections.

- `workflow.py` builds every data and analysis cache required by the figures.
- `visualization.ipynb` previews each panel and renders one main and three
  supplementary figures.
- `config.py` defines the section cohorts and analysis parameters.

## How to Run

Run from the project root with the `sonic_paper` environment from
[`environment.yml`](../../environment.yml):

```bash
conda run -n sonic_paper \
  python -m scripts.embryo_xspecies.workflow --progress
```

A normal run validates each cache and skips current work. `--force` rebuilds
derived caches without touching raw h5ad files. The separate
`--force-download` option re-downloads every configured section and rebuilds
all dependent caches; use it only for a complete raw-data refresh.

The workflow runs seven stages in dependency order:

1. Download the configured MOSTA and HESTA sections and update species metadata.
2. Prepare observed-bin SpatialData zarrs and species-level gene QC.
3. Build or validate the strict MGI human-mouse 1:1 orthology table.
4. Cache per-section SONIC spatial-variability tests for all eligible genes.
5. Cache per-section radial spectra on one shared frequency grid and fit joint PCA.
6. Cache embedding annotations, clustering, GO enrichment, and local mixing.
7. Cache global temporal variation and consecutive-stage expression,
   PCA-movement, DE, DF, DF-norm, and GO metrics.

The two downloaders, `preprocessing.py`, `orthology.py`, `sv.py`, and
`embeddings.py` retain standalone CLIs for targeted or interrupted-stage work.
Embedding and temporal metrics are workflow-only because both consume the
same in-memory joint PCA.

## Cached Outputs

- `data/embryo_xspecies/{mosta,hesta}/raw/`: downloaded per-section h5ad files.
- `data/embryo_xspecies/{mosta,hesta}/sdata/`: prepared SpatialData zarrs.
- `results/embryo_xspecies/{mouse_only,human_only}/`: preprocessing QC,
  per-section SV tables, and per-section spectrum caches.
- `results/embryo_xspecies/comparison/human_mouse_orthologs.tsv`: strict 1:1
  ortholog and protein-coding annotations.
- `results/embryo_xspecies/comparison/metrics/`: embedding cluster, GO, and
  local-mixing tables.
- `results/embryo_xspecies/comparison/temporal/`: global variation,
  conservation, selected genes, per-gene stage transitions, and GO results.

PCA coordinates and the long per-gene-section table are reconstructed in
memory and are not cached.

## Rendering Figures

After the workflow completes, execute the notebook:

```bash
conda run -n sonic_paper python -m jupyter nbconvert --to notebook --execute \
  scripts/embryo_xspecies/visualization.ipynb \
  --output /private/tmp/embryo_xspecies_visualization.executed.ipynb \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=sonic_paper
```

The notebook previews every panel before saving flat under
`figures/embryo_xspecies/`. Previews remain in the notebook; only PDFs are
written to disk:

- Main: `embedding.pdf`.
- Supplementary: `cluster_go.pdf`.
- Supplementary: `temporal_variation.pdf`.
- Supplementary: `stage_transitions.pdf`.
