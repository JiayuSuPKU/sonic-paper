# SONIC paper analyses

Analysis scripts for the paper [Consistent and scalable detection and comparison of spatial patterns](https://arxiv.org/html/2602.02825v1).

SONIC stands for **Spatial Organization through Nonrandom-pattern Inference and
Comparison** and was formerly released as QuadSV.

The reusable core package now lives in the separate
[`JiayuSuPKU/sonic`](https://github.com/JiayuSuPKU/sonic) repository.

## Reproducible environment

Create the paper environment from the repository root:

```bash
mamba env create -f environment.yml
conda run -n sonic_paper python -c "import sonic; print(sonic.__version__)"
```

The environment pins the published `sonic-spatial==1.0.0rc1` distribution;
Python code imports it as `sonic`. For active package development, replace the
published installation with the sibling source checkout:

```bash
conda run -n sonic_paper python -m pip install --no-build-isolation -e ../sonic
```
