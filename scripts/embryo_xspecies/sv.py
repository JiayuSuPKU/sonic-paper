"""Cache per-section SONIC spatial-variability tests for all eligible genes.

Each species cache contains one compressed table per configured section. The
feature universe is always the complete panel returned by
``utils.select_gene_panel``; callers cannot substitute an ortholog or marker
subset. Raw Q, Z, and p-values are retained without FDR filtering.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from . import utils

LOGGER = logging.getLogger("embryo_xspecies.sv")
RESULT_COLUMNS = (
    "species",
    "section_id",
    "timepoint",
    "feature",
    "Q",
    "Z_score",
    "P_value",
)
STAT_COLUMNS = ("Q", "Z_score", "P_value")


def _feature_hash(features: Sequence[str]) -> str:
    """Hash feature membership independently of display/ranking order."""
    return hashlib.sha256("\n".join(sorted(features)).encode()).hexdigest()


def _analysis_config() -> dict[str, Any]:
    import sonic

    return {
        "schema_version": 2,
        "expression_mode": C.ANALYSIS_EXPRESSION_MODE,
        "normalization_target_sum": C.ANALYSIS_NORMALIZATION_TARGET_SUM,
        "sonic_version": sonic.__version__,
        "detector": {
            "kernel_method": C.SV_KERNEL_METHOD,
            "rho": C.SV_RHO,
            "neighbor_degree": C.SV_NEIGHBOR_DEGREE,
            "spacing_um": [C.BIN_SIZE_UM, C.BIN_SIZE_UM],
            "fft_solver": C.SV_FFT_SOLVER,
        },
    }


def _manifest(cfg: Any, features: list[str], metadata: pd.DataFrame) -> dict[str, Any]:
    gene_qc = Path(cfg.RESULT_DIR) / "gene_qc.csv"
    if not gene_qc.exists():
        raise FileNotFoundError(f"Missing {gene_qc}; run preprocessing first.")
    sections = []
    for row in metadata.itertuples(index=False):
        zarr = cfg.SDATA_DIR / f"{row.section_id}.zarr"
        if not zarr.exists():
            raise FileNotFoundError(f"Missing {zarr}; run preprocessing first.")
        sections.append(
            {
                "section_id": str(row.section_id),
                "timepoint": str(row.timepoint),
                "prepared_zarr_mtime_ns": zarr.stat().st_mtime_ns,
            }
        )
    return {
        **_analysis_config(),
        "species": cfg.PANEL_KEY,
        "feature_scope": "all genes returned by utils.select_gene_panel",
        "min_gene_detected_bin_fraction": C.MIN_GENE_DETECTED_BIN_FRACTION,
        "n_features": len(features),
        "feature_sha256": _feature_hash(features),
        "gene_qc_sha256": hashlib.sha256(gene_qc.read_bytes()).hexdigest(),
        "sections": sections,
        "artifact_pattern": "<section_id>.csv.gz",
        "fdr_filter_applied": False,
    }


def _slide_path(cfg: Any, section_id: str) -> Path:
    return cfg.SV_RESULT_DIR / f"{section_id}.csv.gz"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _validate_manifest(cfg: Any, expected: dict[str, Any]) -> None:
    if not cfg.SV_MANIFEST_JSON.exists():
        raise FileNotFoundError(
            f"Missing {cfg.SV_MANIFEST_JSON}; run `python -m scripts.embryo_xspecies.sv`."
        )
    if json.loads(cfg.SV_MANIFEST_JSON.read_text()) != expected:
        raise RuntimeError(
            f"{cfg.SV_MANIFEST_JSON} does not match the current SV analysis; "
            "rebuild with `python -m scripts.embryo_xspecies.sv --force`."
        )


def _validate_inventory(cfg: Any, metadata: pd.DataFrame) -> None:
    expected = {cfg.SV_MANIFEST_JSON}
    expected.update(_slide_path(cfg, str(section_id)) for section_id in metadata["section_id"])
    unexpected = sorted(path.name for path in cfg.SV_RESULT_DIR.iterdir() if path not in expected)
    if unexpected:
        raise RuntimeError(f"Unexpected artifacts in {cfg.SV_RESULT_DIR}: {unexpected[:10]}")


def _validate_frame(
    frame: pd.DataFrame,
    *,
    source: str | Path,
    species: str,
    section_id: str,
    timepoint: str,
    expected_features: list[str],
) -> pd.DataFrame:
    if list(frame.columns) != list(RESULT_COLUMNS):
        raise ValueError(
            f"{source} columns are {list(frame.columns)}; expected {list(RESULT_COLUMNS)}."
        )
    frame["feature"] = frame["feature"].astype(str)
    if frame["feature"].duplicated().any():
        raise ValueError(f"{source} contains duplicate features.")
    observed = set(frame["feature"])
    expected = set(expected_features)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{source} has the wrong feature universe; missing={missing[:5]}, extra={extra[:5]}."
        )
    if len(frame) != len(expected_features):
        raise ValueError(f"{source} has {len(frame)} rows; expected {len(expected_features)}.")
    identities = {
        "species": species,
        "section_id": section_id,
        "timepoint": timepoint,
    }
    for column, value in identities.items():
        if not frame[column].astype(str).eq(value).all():
            raise ValueError(f"{source} has inconsistent {column!r} values.")
    statistics = frame.loc[:, STAT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(statistics).all():
        raise ValueError(f"{source} contains non-finite SONIC statistics.")
    if not frame["P_value"].between(0.0, 1.0).all():
        raise ValueError(f"{source} contains p-values outside [0, 1].")
    return frame.set_index("feature").loc[expected_features].reset_index()[list(RESULT_COLUMNS)]


def _load_slide(
    cfg: Any,
    section_id: str,
    timepoint: str,
    cached_features: list[str],
    requested_features: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single section's SV results from the species cache."""
    path = _slide_path(cfg, section_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run `python -m scripts.embryo_xspecies.sv`.")
    frame = _validate_frame(
        pd.read_csv(path),
        source=path,
        species=cfg.PANEL_KEY,
        section_id=section_id,
        timepoint=timepoint,
        expected_features=cached_features,
    )
    if requested_features is None:
        return frame
    if len(requested_features) != len(set(requested_features)):
        raise ValueError("Requested SV features contain duplicates.")
    missing = sorted(set(requested_features) - set(cached_features))
    if missing:
        raise ValueError(
            f"Requested genes are outside the {cfg.PANEL_KEY} SV panel: {missing[:10]}"
        )
    return frame.set_index("feature").loc[requested_features].reset_index()


def _compute_slide(
    cfg: Any,
    section_id: str,
    timepoint: str,
    features: list[str],
) -> pd.DataFrame:
    """Compute a single section's SV results using sonic Q-test."""
    from sonic import DetectorGrid

    prepared, _, panel, _ = utils.load_prepared_sections(
        cfg,
        section_ids=[section_id],
        gene_panel=features,
        expression_mode=C.ANALYSIS_EXPRESSION_MODE,
        normalization_target_sum=C.ANALYSIS_NORMALIZATION_TARGET_SUM,
        include_domain_channels=False,
    )
    detector = DetectorGrid(
        kernel_method=C.SV_KERNEL_METHOD,
        rho=C.SV_RHO,
        neighbor_degree=C.SV_NEIGHBOR_DEGREE,
        spacing=(C.BIN_SIZE_UM, C.BIN_SIZE_UM),
        fft_solver=C.SV_FFT_SOLVER,
        workers=C.SV_FFT_WORKERS,
    ).setup_data(
        prepared[0],
        bins="bins",
        table_name="bins_table",
        col_key="array_col",
        row_key="array_row",
    )
    result = detector.compute_qstat(
        features=list(panel),
        n_jobs=C.SV_N_JOBS,
        workers=C.SV_FFT_WORKERS,
        return_pval=True,
        chunk_size=C.SV_CHUNK_SIZE,
        show_progress=False,
    ).reset_index(names="feature")
    result = result[["feature", "Q", "Z_score", "P_value"]]
    result.insert(0, "timepoint", timepoint)
    result.insert(0, "section_id", section_id)
    result.insert(0, "species", cfg.PANEL_KEY)
    result = _validate_frame(
        result,
        source=f"computed {cfg.PANEL_KEY} {section_id}",
        species=cfg.PANEL_KEY,
        section_id=section_id,
        timepoint=timepoint,
        expected_features=features,
    )
    del detector, prepared
    gc.collect()
    return result


def _run_species(cfg: Any, *, force: bool) -> dict[str, int]:
    """Run SV test for all sections of a single species, caching results."""
    features = utils.select_gene_panel(cfg)
    metadata = utils.filtered_metadata(cfg)
    expected_manifest = _manifest(cfg, features, metadata)
    if force and cfg.SV_RESULT_DIR.exists():
        shutil.rmtree(cfg.SV_RESULT_DIR)
    cfg.SV_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if cfg.SV_MANIFEST_JSON.exists():
        _validate_manifest(cfg, expected_manifest)
    else:
        existing = sorted(path.name for path in cfg.SV_RESULT_DIR.iterdir())
        if existing:
            raise RuntimeError(
                f"Missing manifest for existing artifacts in {cfg.SV_RESULT_DIR}: {existing[:10]}"
            )
        _write_json(cfg.SV_MANIFEST_JSON, expected_manifest)
    _validate_inventory(cfg, metadata)

    for row in metadata.itertuples(index=False):
        section_id = str(row.section_id)
        timepoint = str(row.timepoint)
        path = _slide_path(cfg, section_id)
        if path.exists():
            _load_slide(cfg, section_id, timepoint, features)
            continue
        LOGGER.info(
            "SONIC spatial variability: %s %s (%d genes)",
            cfg.PANEL_KEY,
            section_id,
            len(features),
        )
        result = _compute_slide(cfg, section_id, timepoint, features)
        temporary = path.with_name(f"{path.name}.tmp")
        result.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        del result
        gc.collect()

    return {
        "n_sections": len(metadata),
        "n_features": len(features),
        "n_rows": len(metadata) * len(features),
    }


def run(
    species: Sequence[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, dict[str, int]]:
    """Compute or validate complete species-wide SV caches."""
    names = list(C.SPECIES_CONFIGS) if species is None else list(species)
    if not names:
        raise ValueError("At least one species is required.")
    if len(names) != len(set(names)):
        raise ValueError("Species list contains duplicates.")
    unknown = sorted(set(names) - set(C.SPECIES_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown species: {unknown}")
    return {name: _run_species(C.SPECIES_CONFIGS[name], force=force) for name in names}


def load(
    panels: dict[str, list[str]],
    *,
    section_ids: dict[str, list[str] | None] | None = None,
) -> pd.DataFrame:
    """Load requested genes from validated complete species-wide SV caches."""
    unknown = sorted(set(panels) - set(C.SPECIES_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown species: {unknown}")
    if section_ids is not None and set(section_ids) != set(panels):
        raise ValueError("section_ids keys must exactly match the requested panels.")
    frames: list[pd.DataFrame] = []
    for species, requested_features in panels.items():
        if not requested_features:
            raise ValueError(f"No SV features requested for {species!r}.")
        cfg = C.SPECIES_CONFIGS[species]
        cached_features = utils.select_gene_panel(cfg)
        all_metadata = utils.filtered_metadata(cfg)
        _validate_manifest(cfg, _manifest(cfg, cached_features, all_metadata))
        _validate_inventory(cfg, all_metadata)
        selected_ids = None if section_ids is None else section_ids[species]
        metadata = utils.filtered_metadata(cfg, section_ids=selected_ids)
        frames.extend(
            _load_slide(
                cfg,
                str(row.section_id),
                str(row.timepoint),
                cached_features,
                requested_features,
            )
            for row in metadata.itertuples(index=False)
        )
    if not frames:
        raise ValueError("No species panels were requested.")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--species",
        action="append",
        choices=sorted(C.SPECIES_CONFIGS),
        help="Species to run; repeat as needed. Default: both.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the selected species caches.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summaries = run(args.species, force=args.force)
    for species, summary in summaries.items():
        LOGGER.info(
            "%s SV cache: %d sections x %d genes = %d rows",
            species,
            summary["n_sections"],
            summary["n_features"],
            summary["n_rows"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
