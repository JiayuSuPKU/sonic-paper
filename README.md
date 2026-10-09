# SONIC paper analyses

Analysis scripts for the paper [Consistent and scalable detection and comparison of spatial patterns](https://arxiv.org/html/2602.02825v1).

SONIC stands for **Spatial Organization through Nonrandom-pattern Inference and
Comparison** and was formerly released as QuadSV.

The reusable core package now lives in the separate
[`JiayuSuPKU/sonic`](https://github.com/JiayuSuPKU/sonic) repository.

## Analyses

| Workflow | Scope |
| --- | --- |
| [Q-test benchmarks](scripts/qtest_benchmark/) | Synthetic validation, runtime scaling, mouse intestine, and tumor lineages |
| [Pattern-comparison benchmarks](scripts/comparator_benchmark/) | Alzheimer's disease and postnatal mouse brain development |
| [Glioma](scripts/glioma_idh/) | IDH-associated differences in spatial expression |
| [Embryonic atlas](scripts/embryo_xspecies/) | Mouse and human developmental trajectories and pattern retrieval |
| [Visium HD SPLISOSM](scripts/splisosm_visiumhd_cbs/) | Spatial isoform usage and regulator associations across assays |
| [Atera SPLISOSM](scripts/splisosm_atera_brca/) | Breast-cancer transcript usage and ANKRD30A validation |

Each analysis directory documents its inputs, configuration, and commands.
Workflows build result caches; `visualization.ipynb` notebooks generate figures
from those caches.

## Reproducible environment

Create the paper environment from the repository root:

```bash
mamba env create -f environment.yml
conda run -n sonic_paper python -c "import sonic; print(sonic.__version__)"
```

The environment pins the published `sonic-spatial==1.0.0rc2` distribution;
Python code imports it as `sonic`. For active package development, replace the
published installation with the sibling source checkout:

```bash
conda run -n sonic_paper python -m pip install --no-build-isolation -e ../sonic
```

The two SPLISOSM workflows require the separate `sonic_paper_splisosm`
environment. Follow the setup instructions in the
[Atera README](scripts/splisosm_atera_brca/README.md#minimal-dependencies)
before running those analyses.

## Data and outputs

Download or prepare the inputs described in each analysis README before running
its workflow. Check `config.py` for input and reference paths, especially the
Visium HD preprocessed inputs and the reference files used by SPLISOSM.
Controlled-access TCGA analyses require separate GDC authorization and a local
token file.

Data, generated results and figures are excluded from Git. Notebooks are stored
without execution outputs. Manuscript drafts and co-author review packages are
maintained separately from this analysis repository.

## Run analysis caches

Inspect the commands first, then run the selected workflows from the repository
root:

```bash
conda run -n sonic_paper python -m scripts.run_all_analyses --dry-run
conda run -n sonic_paper python -m scripts.run_all_analyses comparator_benchmark
```

Omit the workflow names to run all six analyses in their documented environments.
Pass `--force` only when the selected workflows should recompute their derived
caches.

## License

This repository is licensed under the [BSD 3-Clause License](LICENSE).
