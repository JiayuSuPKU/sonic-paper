"""Compute and cache per-gene-slide binned spectra.

Genes are selected by :func:`utils.select_gene_panel`, which includes every named,
non-lncRNA, non-mitochondrial gene detected in more than the configured
fraction of bins in every slide of its species. Each section persists one
``<section_id>.npz`` containing fitted spectra and provenance on the consensus
frequency grid shared by every configured mouse and human section. Downstream
callers may fit PCA using all gene-slide spectra and project binary domain masks
into that gene-defined space; domain masks do not influence the PCA fit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import config as C
from . import utils

LOGGER = logging.getLogger("embryo_xspecies.embeddings")
_EPS = 1e-12
_CACHE_KEYS = {
    "manifest_json",
    "spectra",
    "dc",
    "presence",
    "feature_names",
    "section_id",
    "freq_edges",
    "frequency_bin_intervals",
}


# ---------------------------------------------------------------------------
# Spectral computation and cache
# ---------------------------------------------------------------------------
def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _section_path(cfg: Any, section_id: str) -> Path:
    return cfg.SPECTRA_RESULT_DIR / f"{section_id}.npz"


def _section_manifest(
    cfg: Any,
    features: list[str],
    genes: list[str],
    metadata_row: Any,
    freq_edges: np.ndarray,
) -> dict[str, Any]:
    import quadsv

    # Each section's npz will include its own manifest json
    gene_qc = Path(cfg.RESULT_DIR) / "gene_qc.csv"
    section_id = str(metadata_row.section_id)
    zarr = cfg.SDATA_DIR / f"{section_id}.zarr"
    return {
        "schema_version": 2,
        "species": cfg.PANEL_KEY,
        "feature_scope": "all genes returned by utils.select_gene_panel plus domain masks",
        "expression_mode": C.ANALYSIS_EXPRESSION_MODE,
        "normalization_target_sum": C.ANALYSIS_NORMALIZATION_TARGET_SUM,
        "min_gene_detected_bin_fraction": C.MIN_GENE_DETECTED_BIN_FRACTION,
        "background_normalized": True,
        "quadsv_version": quadsv.__version__,
        "bin_size_um": C.BIN_SIZE_UM,
        "n_radial_bins_requested": C.N_RADIAL_BINS,
        "frequency_spacing": C.FREQ_SPACING,
        "frequency_edges": [float(value) for value in freq_edges],
        "n_genes": len(genes),
        "n_features": len(features),
        "feature_sha256": _sha256_lines(features),
        "gene_qc_sha256": hashlib.sha256(gene_qc.read_bytes()).hexdigest(),
        "section": {
            "section_id": section_id,
            "timepoint": str(metadata_row.timepoint),
            "prepared_zarr_mtime_ns": zarr.stat().st_mtime_ns,
        },
        "artifact": f"{section_id}.npz",
    }


def _validate_inventory(cfg: Any, section_ids: list[str]) -> None:
    expected = {_section_path(cfg, section_id) for section_id in section_ids}
    unexpected = sorted(
        path.name for path in cfg.SPECTRA_RESULT_DIR.iterdir() if path not in expected
    )
    if unexpected:
        raise RuntimeError(
            f"Unexpected artifacts in {cfg.SPECTRA_RESULT_DIR}: {unexpected[:10]}. "
            "Rebuild with `python -m scripts.embryo_xspecies.embeddings --force`."
        )


def _load_section_cache(
    path: Path,
    *,
    expected_features: list[str],
    expected_section: str,
    expected_edges: np.ndarray,
    expected_manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cached:
        if set(cached.files) != _CACHE_KEYS:
            raise ValueError(
                f"{path} arrays are {sorted(cached.files)}; "
                f"expected {sorted(_CACHE_KEYS)}."
            )
        arrays = {key: cached[key] for key in cached.files}

    if json.loads(str(arrays["manifest_json"].item())) != expected_manifest:
        raise RuntimeError(
            f"{path} does not match the current analysis; rebuild with "
            "`python -m scripts.embryo_xspecies.embeddings --force`."
        )
    features = arrays["feature_names"].astype(str).tolist()
    if features != expected_features:
        raise ValueError(f"{path} has the wrong feature order.")
    if str(arrays["section_id"].item()) != expected_section:
        raise ValueError(f"{path} has the wrong section ID.")
    if not np.array_equal(arrays["freq_edges"], expected_edges):
        raise ValueError(f"{path} has the wrong consensus frequency grid.")
    expected_intervals = np.column_stack([expected_edges[:-1], expected_edges[1:]])
    if not np.array_equal(arrays["frequency_bin_intervals"], expected_intervals):
        raise ValueError(f"{path} has the wrong frequency intervals.")

    expected_shape = (len(expected_features), expected_intervals.shape[0])
    if arrays["spectra"].shape != expected_shape:
        raise ValueError(
            f"{path} spectra shape is {arrays['spectra'].shape}; "
            f"expected {expected_shape}."
        )
    scalar_shape = expected_shape[:1]
    if arrays["dc"].shape != scalar_shape or arrays["presence"].shape != scalar_shape:
        raise ValueError(f"{path} has invalid DC or presence dimensions.")
    if not np.isfinite(arrays["spectra"]).all() or not np.isfinite(arrays["dc"]).all():
        raise ValueError(f"{path} contains non-finite fitted values.")
    return arrays


def _write_section_cache(
    path: Path,
    comparator: Any,
    section_id: str,
    manifest: dict[str, Any],
) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
            spectra=np.asarray(comparator.spectra_[0], dtype=np.float32),
            dc=np.asarray(comparator.dc_[0], dtype=np.float32),
            presence=np.asarray(comparator.presence_[0], dtype=bool),
            feature_names=np.asarray(comparator.gene_names, dtype=str),
            section_id=np.asarray(section_id),
            freq_edges=np.asarray(comparator.freq_edges, dtype=np.float64),
            frequency_bin_intervals=np.asarray(
                comparator.frequency_bin_intervals_, dtype=np.float64
            ),
        )
    temporary.replace(path)


def _cache_species_sections(
    cfg: Any,
    *,
    freq_edges: np.ndarray,
    force: bool,
    progress: bool,
) -> tuple[Any, list[str], list[str]]:
    """Compute or validate one species' independent section caches."""
    if force and cfg.SPECTRA_RESULT_DIR.exists():
        shutil.rmtree(cfg.SPECTRA_RESULT_DIR)
    cfg.SPECTRA_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    genes = utils.select_gene_panel(cfg)
    metadata = utils.filtered_metadata(cfg)
    domain_labels = utils.prepared_domain_labels(cfg)
    domain_channels = [f"{C.DOMAIN_CHANNEL_PREFIX}{label}" for label in domain_labels]
    features = genes + domain_channels
    section_ids = metadata["section_id"].astype(str).tolist()
    _validate_inventory(cfg, section_ids)

    n_sections = len(section_ids)
    n_features = len(features)
    n_frequencies = len(freq_edges) - 1
    expected_intervals = np.column_stack([freq_edges[:-1], freq_edges[1:]])

    for index, row in enumerate(metadata.itertuples(index=False)):
        section_id = str(row.section_id)
        path = _section_path(cfg, section_id)
        manifest = _section_manifest(cfg, features, genes, row, freq_edges)
        if path.exists():
            _load_section_cache(
                path,
                expected_features=features,
                expected_section=section_id,
                expected_edges=freq_edges,
                expected_manifest=manifest,
            )
            continue

        LOGGER.info(
            "%s spectra: %s (%d/%d; %d genes + %d domains)",
            cfg.PANEL_KEY,
            section_id,
            index + 1,
            n_sections,
            len(genes),
            len(domain_channels),
        )
        sections, _, panel, loaded_domain_channels = utils.load_prepared_sections(
            cfg,
            section_ids=[section_id],
            gene_panel=genes,
            expression_mode=C.ANALYSIS_EXPRESSION_MODE,
            normalization_target_sum=C.ANALYSIS_NORMALIZATION_TARGET_SUM,
            domain_labels=domain_labels,
        )
        if panel != genes or loaded_domain_channels != domain_channels:
            raise RuntimeError(f"{cfg.PANEL_KEY} {section_id} feature order changed.")
        section_comparator = utils.build_comparator(
            sections,
            features,
            freq_edges=freq_edges,
            fft_chunk_size="auto",
            adaptive_freq_edges=False,
        )
        section_comparator.compute_spectra(n_jobs=1, progress=progress)
        section_comparator.normalize_background()
        section_comparator.spectra_ = np.asarray(
            section_comparator.spectra_, dtype=np.float32
        )
        section_comparator.dc_ = np.asarray(section_comparator.dc_, dtype=np.float32)
        section_comparator.presence_ = np.asarray(
            section_comparator.presence_, dtype=bool
        )
        expected_shape = (1, n_features, n_frequencies)
        if section_comparator.spectra_.shape != expected_shape:
            raise RuntimeError(
                f"{cfg.PANEL_KEY} {section_id} spectra shape is "
                f"{section_comparator.spectra_.shape}; expected {expected_shape}."
            )
        if not np.array_equal(
            section_comparator.frequency_bin_intervals_, expected_intervals
        ):
            raise RuntimeError(
                f"{cfg.PANEL_KEY} {section_id} changed the consensus frequency grid."
            )
        _write_section_cache(path, section_comparator, section_id, manifest)
        del section_comparator, sections
        gc.collect()

    return metadata, genes, domain_labels


def _load_species_arrays(
    cfg: Any,
    *,
    freq_edges: np.ndarray,
    metadata: Any,
    genes: list[str],
    domain_labels: list[str],
) -> dict[str, Any]:
    """Load one species' validated section caches into aligned arrays."""
    domain_channels = [f"{C.DOMAIN_CHANNEL_PREFIX}{label}" for label in domain_labels]
    features = genes + domain_channels
    section_ids = metadata["section_id"].astype(str).tolist()
    n_sections = len(section_ids)
    n_features = len(features)
    n_frequencies = len(freq_edges) - 1
    spectra = np.empty((n_sections, n_features, n_frequencies), dtype=np.float32)
    dc = np.empty((n_sections, n_features), dtype=np.float32)
    presence = np.empty((n_sections, n_features), dtype=bool)

    for index, row in enumerate(metadata.itertuples(index=False)):
        section_id = str(row.section_id)
        arrays = _load_section_cache(
            _section_path(cfg, section_id),
            expected_features=features,
            expected_section=section_id,
            expected_edges=freq_edges,
            expected_manifest=_section_manifest(cfg, features, genes, row, freq_edges),
        )
        spectra[index] = arrays["spectra"]
        dc[index] = arrays["dc"]
        presence[index] = arrays["presence"]

    gene_cols = np.arange(len(genes), dtype=int)
    domain_cols = np.arange(len(genes), len(features), dtype=int)
    return {
        "label": cfg.PANEL_KEY,
        "cfg": cfg,
        "section_ids": section_ids,
        "meta": metadata,
        "spectra": spectra,
        "dc": dc,
        "presence": presence,
        "freq_edges": freq_edges,
        "gene_cols": gene_cols,
        "gene_names": genes,
        "name_to_col": {name: i for i, name in enumerate(genes)},
        "domain_cols": domain_cols,
        "domain_labels": [
            name.removeprefix(C.DOMAIN_CHANNEL_PREFIX) for name in domain_channels
        ],
    }


def comparator_from_cached_spectra(
    bundle: dict[str, Any],
    section_indices: Sequence[int] | np.ndarray,
    gene_names: Sequence[str],
) -> Any:
    """Restore a fitted comparator view for selected sections and genes.

    Prepared expression is loaded only for the requested sections so QuadSV's
    public differential-test methods can operate on the cached spectral state
    without recomputing FFTs or materializing every section at once.
    """
    indices = np.asarray(section_indices, dtype=int)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("section_indices must be a non-empty one-dimensional array.")
    if np.any(indices < 0) or np.any(indices >= len(bundle["section_ids"])):
        raise IndexError("section_indices contains an out-of-range section.")
    genes = list(map(str, gene_names))
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("gene_names must be non-empty and unique.")
    missing = sorted(set(genes) - set(bundle["name_to_col"]))
    if missing:
        raise KeyError(f"Genes are absent from the spectra cache: {missing[:5]}")

    section_ids = [bundle["section_ids"][index] for index in indices]
    sections, metadata, loaded_genes, domain_channels = utils.load_prepared_sections(
        bundle["cfg"],
        section_ids=section_ids,
        gene_panel=genes,
        expression_mode=C.ANALYSIS_EXPRESSION_MODE,
        normalization_target_sum=C.ANALYSIS_NORMALIZATION_TARGET_SUM,
        include_domain_channels=False,
    )
    if metadata["section_id"].astype(str).tolist() != section_ids:
        raise RuntimeError("Prepared sections do not match the cached section order.")
    if loaded_genes != genes or domain_channels:
        raise RuntimeError("Prepared comparator feature order changed.")

    comparator = utils.build_comparator(
        sections,
        genes,
        freq_edges=np.asarray(bundle["freq_edges"]),
        adaptive_freq_edges=False,
    )
    columns = np.asarray([bundle["name_to_col"][gene] for gene in genes], dtype=int)
    comparator.spectra_ = bundle["spectra"][indices][:, columns, :]
    comparator.dc_ = bundle["dc"][indices][:, columns]
    comparator.presence_ = bundle["presence"][indices][:, columns]
    comparator.frequency_bin_intervals_ = np.column_stack(
        [bundle["freq_edges"][:-1], bundle["freq_edges"][1:]]
    )
    return comparator


# ---------------------------------------------------------------------------
# PCA embedding
# ---------------------------------------------------------------------------
def frequency_weights(
    n_frequencies: int,
    custom_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return equal or explicitly supplied mean-one frequency weights."""
    if n_frequencies < 1:
        raise ValueError("n_frequencies must be positive.")
    if custom_weights is None:
        return np.ones(n_frequencies, dtype=np.float64)
    weights = np.asarray(custom_weights, dtype=np.float64)
    if weights.shape != (n_frequencies,):
        raise ValueError(
            f"custom_weights must have shape ({n_frequencies},), got {weights.shape}."
        )
    mean = float(weights.mean())
    if not np.isfinite(weights).all() or np.any(weights < 0) or mean == 0:
        raise ValueError("custom_weights must be finite, non-negative, and nonzero.")
    return weights / mean


def fit_shape_embedding(
    fit_rows: np.ndarray,
    *,
    n_components: int = 2,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[PCA, Callable[[np.ndarray], np.ndarray]]:
    """Fit PCA on L1-log, standardized, optionally weighted spectra.

    The returned transform applies the training mean, standard deviation, and
    frequency weights to new spectra before projecting them with the fitted PCA.
    """

    def shape_features(rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        probabilities = rows / np.maximum(rows.sum(axis=1, keepdims=True), _EPS)
        return np.log(np.maximum(probabilities, _EPS))

    features = shape_features(fit_rows)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + _EPS
    bin_weights = frequency_weights(features.shape[1], weights)[None, :]
    pca = PCA(n_components=n_components, random_state=0).fit(
        ((features - mean) / std) * bin_weights
    )

    def transform(rows: np.ndarray) -> np.ndarray:
        transformed = shape_features(rows)
        return pca.transform(((transformed - mean) / std) * bin_weights)

    return pca, transform


def _fit_joint_embedding(
    species: dict[str, dict[str, Any]],
) -> tuple[PCA, Callable[[np.ndarray], np.ndarray]]:
    frequency_counts = {bundle["spectra"].shape[-1] for bundle in species.values()}
    if len(frequency_counts) != 1:
        raise RuntimeError(
            f"Species spectra use different frequency counts: {frequency_counts}"
        )
    # Combine all species' gene-slide spectra into one matrix for PCA fitting.
    gene_rows = [
        bundle["spectra"][:, bundle["gene_cols"], :].reshape(
            -1, bundle["spectra"].shape[-1]
        )
        for bundle in species.values()
    ]
    pca, transform = fit_shape_embedding(np.vstack(gene_rows), n_components=C.N_PC)
    # Project each slide's domain masks into the joint PCA space.
    for bundle in species.values():
        shape = bundle["spectra"].shape
        coordinates = transform(bundle["spectra"].reshape(-1, shape[-1])).reshape(
            shape[0], shape[1], C.N_PC
        )
        domain_cols = bundle["domain_cols"]
        if domain_cols.size:
            domain_coordinates = coordinates[:, domain_cols, :]
            domain_coordinates[bundle["dc"][:, domain_cols] == 0] = np.nan
            coordinates[:, domain_cols, :] = domain_coordinates
        bundle["coords"] = coordinates
    return pca, transform


def cache_species_spectra(*, force: bool = False, progress: bool = False) -> np.ndarray:
    """Compute or validate both species' caches with section-bounded memory."""
    C.ensure_output_dirs()
    # Compute the shared frequency grid for all sections.
    freq_edges = utils.shared_supported_freq_edges(list(C.SPECIES_CONFIGS.values()))

    # Cache each species' sections independently, then free memory before the next.
    for cfg in C.SPECIES_CONFIGS.values():
        _cache_species_sections(
            cfg,
            freq_edges=freq_edges,
            force=force,
            progress=progress,
        )
        gc.collect()
    return freq_edges


def load_spectra_and_fit_joint_embedding(
    *, force: bool = False, progress: bool = False
) -> dict[str, Any]:
    """Load or cache spectra, then fit the joint PCA embedding in memory."""
    C.ensure_output_dirs()
    freq_edges = utils.shared_supported_freq_edges(list(C.SPECIES_CONFIGS.values()))
    # Load per-slide cached spectral arrays for each species
    species = {}
    for name, cfg in C.SPECIES_CONFIGS.items():
        metadata, genes, domain_labels = _cache_species_sections(
            cfg,
            freq_edges=freq_edges,
            force=force,
            progress=progress,
        )
        species[name] = _load_species_arrays(
            cfg,
            freq_edges=freq_edges,
            metadata=metadata,
            genes=genes,
            domain_labels=domain_labels,
        )

    # Fit PCA using all species' gene-slide spectra, then project each slide's domain masks.
    # PCA coordinates are stored in place in each species' bundle under the "coords" key.
    pca, transform = _fit_joint_embedding(species)
    n_coordinates = sum(
        bundle["coords"].shape[0] * bundle["coords"].shape[1]
        for bundle in species.values()
    )
    LOGGER.info(
        "Joint PCA: %d in-memory feature-slide coordinates; variance %s",
        n_coordinates,
        ", ".join(
            f"PC{i + 1}={100 * value:.1f}%"
            for i, value in enumerate(pca.explained_variance_ratio_)
        ),
    )
    return {
        "species": species,
        "pca": pca,
        "transform": transform,
        "freq_edges": freq_edges,
    }


def query_nearest_gene_stages(
    query_gene: str,
    query_species: str,
    *,
    query_section_id: str,
    embedding: dict[str, Any] | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """Return the nearest exact gene-section patterns in the joint PCA space.

    The exact query pattern is excluded. ``top_n`` is the total number of
    neighbors across both species, not a per-species limit.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if embedding is None:
        embedding = load_spectra_and_fit_joint_embedding()
    if query_species not in embedding["species"]:
        raise KeyError(f"Unknown query species {query_species!r}.")

    query_bundle = embedding["species"][query_species]
    if query_gene not in query_bundle["name_to_col"]:
        raise KeyError(f"{query_gene!r} is absent from the {query_species} gene panel.")
    query_sections = np.asarray(query_bundle["section_ids"], dtype=str)
    section_indices = np.flatnonzero(query_sections == str(query_section_id))
    if len(section_indices) != 1:
        raise KeyError(
            f"Expected one {query_species} section {query_section_id!r}; "
            f"found {len(section_indices)}."
        )
    query_section_index = int(section_indices[0])
    query_coordinate = query_bundle["coords"][
        query_section_index, query_bundle["name_to_col"][query_gene], :
    ]
    query_stage = str(query_bundle["meta"].iloc[query_section_index]["timepoint"])

    candidates = []
    for target_species, bundle in embedding["species"].items():
        gene_names = np.asarray(bundle["gene_names"], dtype=str)
        coordinates = bundle["coords"][:, bundle["gene_cols"], :]
        distances = np.linalg.norm(coordinates - query_coordinate, axis=2)
        if target_species == query_species:
            query_gene_index = np.flatnonzero(gene_names == query_gene)
            if len(query_gene_index) != 1:
                raise RuntimeError(
                    f"Expected one {query_gene!r} entry in the {query_species} panel."
                )
            distances[query_section_index, int(query_gene_index[0])] = np.inf

        flat = distances.ravel()
        n_hits = min(top_n, int(np.isfinite(flat).sum()))
        order = np.argpartition(flat, n_hits - 1)[:n_hits]
        order = order[np.argsort(flat[order], kind="stable")]
        section_index, gene_index = np.divmod(order, len(gene_names))
        candidates.append(
            pd.DataFrame(
                {
                    "target_species": target_species,
                    "target_gene": gene_names[gene_index],
                    "target_stage": np.asarray(bundle["meta"]["timepoint"], dtype=str)[
                        section_index
                    ],
                    "target_section_id": np.asarray(bundle["section_ids"], dtype=str)[
                        section_index
                    ],
                    "distance": flat[order],
                }
            )
        )
    result = (
        pd.concat(candidates, ignore_index=True)
        .sort_values(
            ["distance", "target_species", "target_gene", "target_section_id"],
            kind="stable",
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "query_section_id", query_section_id)
    result.insert(0, "query_stage", query_stage)
    result.insert(0, "query_gene", query_gene)
    result.insert(0, "query_species", query_species)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild both species' per-section spectrum caches.",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cache_species_spectra(force=args.force, progress=args.progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
