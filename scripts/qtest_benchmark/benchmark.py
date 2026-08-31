"""Runtime, implementation-agreement, calibration, and power benchmarks."""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft
from scipy.fft import prev_fast_len
from scipy.ndimage import map_coordinates
from scipy.special import gamma, kv
from sonic.kernels import MatrixKernel
from sonic.kernels.fft import FFTKernel
from sonic.kernels.nufft import NUFFTKernel
from sonic.statistics import compute_null_params, spatial_q_test
from sonic.utils import compute_torus_distance_matrix

from .config import (
    BENCHMARK_PROJECTION_FILE,
    BENCHMARK_SCALING_FILE,
    BENCHMARK_TARGETS,
    M_FEATURES,
    RUNTIME_MEMORY_BUDGET_MB,
    VALIDATION_RESULT_DIR,
    ensure_output_dirs,
)
from .utils import cache_is_reusable

LOGGER = logging.getLogger(__name__)
CACHE_INPUTS = (Path(__file__), Path(__file__).with_name("config.py"))

VALIDATION_CACHE_FILES = {
    "torus_agreement": "agreement_torus.csv",
    "boundary_agreement": "agreement_open_boundary.csv",
    "smooth_agreement": "agreement_nufft_distance.csv",
    "graph_agreement": "agreement_nufft_graph.csv",
    "fpr": "null_fpr.csv",
    "welch_liu": "null_welch_liu.csv",
    "power_regular": "power_regular.csv",
    "power_irregular": "power_irregular.csv",
    "power_hyperparameter": "power_hyperparameters.csv",
}


@dataclass(frozen=True)
class RuntimeImplementation:
    name: str
    max_n: int
    null_method: str
    kind: str


RUNTIME_IMPLEMENTATIONS = (
    RuntimeImplementation("Matrix-Matern (dense)", 10_000, "welch", "matrix_matern"),
    RuntimeImplementation("Matrix-CAR (sparse)", 1_000_000, "welch", "matrix_car"),
    RuntimeImplementation("Matrix-Moran (sparse)", 6_000_000, "clt", "matrix_moran"),
    RuntimeImplementation("FFT-Matern", 10_000_000, "liu", "fft"),
    RuntimeImplementation("NUFFT-Matern", 10_000_000, "welch", "nufft"),
)

BENCHMARK_SAMPLE_SIZES = (
    500,
    1_000,
    2_000,
    3_000,
    4_000,
    5_000,
    6_500,
    8_000,
    10_000,
    20_000,
    50_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    6_000_000,
    10_000_000,
)

_VALIDATION_KERNEL_PARAMETERS = {
    "gaussian": {"bandwidth": 2.0},
    "matern": {"bandwidth": 2.0, "nu": 1.5},
    "car": {"k_neighbors": 4, "rho": 0.9},
    "moran": {"k_neighbors": 4},
}
_POWER_SIGNAL_TO_NOISE_RATIOS = (
    0.0,
    0.03,
    0.05,
    0.07,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    1.0,
)


def _bounded_fft_grid_shape(n: int) -> tuple[int, int]:
    # Keep the FFT-friendly square grid at or below the requested sample limit.
    side = prev_fast_len(max(int(np.sqrt(n)), 8))
    return side, side


def _regular_grid_coordinates(ny: int, nx: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij"
    )
    return np.column_stack((y.ravel(), x.ravel()))


def _build_runtime_kernel(
    spec: RuntimeImplementation, ny: int, nx: int, coords: np.ndarray
):
    if spec.kind == "matrix_matern":
        return MatrixKernel(coords, method="matern", bandwidth=2.5, nu=1.5)
    if spec.kind == "matrix_car":
        return MatrixKernel(coords, method="car", k_neighbors=4, rho=0.9)
    if spec.kind == "matrix_moran":
        return MatrixKernel(coords, method="moran", k_neighbors=4)
    if spec.kind == "fft":
        return FFTKernel((ny, nx), method="matern", bandwidth=2.5, nu=1.5, workers=1)
    if spec.kind == "nufft":
        return NUFFTKernel(
            coords,
            grid_shape=(ny, nx),
            spacing=(1.0, 1.0),
            method="matern",
            bandwidth=2.5,
            nu=1.5,
            workers=1,
        )
    raise ValueError(f"Unknown implementation kind: {spec.kind}")


def _reshape_fft_values(kernel, values: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return values.reshape(ny, nx, -1) if isinstance(kernel, FFTKernel) else values


def cache_runtime_scaling_results(
    *, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure runtime/memory scaling and cache transcriptome projections."""
    ensure_output_dirs()
    runtime_outputs = (BENCHMARK_SCALING_FILE, BENCHMARK_PROJECTION_FILE)
    if cache_is_reusable(
        *runtime_outputs,
        force=force,
        inputs=CACHE_INPUTS,
        logger=LOGGER,
        label="Runtime benchmark",
    ):
        return pd.read_csv(BENCHMARK_SCALING_FILE), pd.read_csv(
            BENCHMARK_PROJECTION_FILE
        )

    rows: list[dict] = []
    rng = np.random.default_rng(42)
    for spec in RUNTIME_IMPLEMENTATIONS:
        for requested_n in BENCHMARK_SAMPLE_SIZES:
            if requested_n > spec.max_n:
                continue
            # The actual regular grid is FFT-friendly and never exceeds max_n.
            ny, nx = _bounded_fft_grid_shape(requested_n)
            n = ny * nx
            coords = _regular_grid_coordinates(ny, nx)
            x1 = rng.standard_normal((n, 1))

            tracemalloc.start()
            start = time.perf_counter()
            try:
                kernel = _build_runtime_kernel(spec, ny, nx, coords)
                null = compute_null_params(kernel, method=spec.null_method)
                build_s = time.perf_counter() - start
                # Single-call cost includes construction plus one tested feature.
                spatial_q_test(
                    _reshape_fft_values(kernel, x1, ny, nx),
                    kernel,
                    null_params=null,
                    return_pval=True,
                    chunk_size=-1,
                )
                single_s = time.perf_counter() - start
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                batch_size = 8
                xb = rng.standard_normal((n, batch_size))
                # A small batch isolates steady-state per-feature test cost.
                start_batch = time.perf_counter()
                spatial_q_test(
                    _reshape_fft_values(kernel, xb, ny, nx),
                    kernel,
                    null_params=null,
                    return_pval=True,
                    chunk_size=-1,
                )
                batch_s = time.perf_counter() - start_batch
                rows.append(
                    {
                        "impl": spec.name,
                        "n": n,
                        "build_s": build_s,
                        "single_call_s": single_s,
                        "peak_mb": peak / 1e6,
                        "batch_features": batch_size,
                        "batch_test_s": batch_s,
                        "per_feature_s": batch_s / batch_size,
                    }
                )
            except (MemoryError, RuntimeError, ValueError) as exc:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                LOGGER.warning(
                    "%s benchmark failed at n=%d: %s: %s",
                    spec.name,
                    n,
                    type(exc).__name__,
                    exc,
                )
                rows.append(
                    {
                        "impl": spec.name,
                        "n": n,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                del coords, x1
                gc.collect()

    benchmark = pd.DataFrame(rows)
    benchmark_cache = benchmark[["impl", "n", "single_call_s", "peak_mb"]]
    benchmark_cache.to_csv(BENCHMARK_SCALING_FILE, index=False)
    projection = _project_transcriptome_runtime(benchmark)
    projection_cache = projection[["impl", "target", "n", "supported", "total_s"]]
    projection_cache.to_csv(BENCHMARK_PROJECTION_FILE, index=False)
    LOGGER.info("Runtime benchmark: cached %d measurements", len(benchmark_cache))
    return benchmark_cache, projection_cache


def _loglog_predict(target_n: float, x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[valid], y[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 2:
        return float("nan")
    # Extrapolate from the largest measurements, where asymptotic scaling dominates.
    use = slice(max(0, len(x) - 3), len(x))
    slope, intercept = np.polyfit(np.log(x[use]), np.log(y[use]), 1)
    return float(np.exp(intercept + slope * np.log(target_n)))


def _project_transcriptome_runtime(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Project build plus 10,000-feature test time from measured asymptotic points."""
    rows: list[dict] = []
    for spec in RUNTIME_IMPLEMENTATIONS:
        sub = benchmark[benchmark["impl"].eq(spec.name)].dropna(
            subset=["n", "build_s", "per_feature_s"]
        )
        for target, n, geometry in BENCHMARK_TARGETS:
            geometry_ok = not (geometry == "irregular" and spec.kind == "fft")
            peak_mb = _loglog_predict(n, sub["n"].to_numpy(), sub["peak_mb"].to_numpy())
            supported = (
                geometry_ok and n <= spec.max_n and peak_mb <= RUNTIME_MEMORY_BUDGET_MB
            )
            build_s = _loglog_predict(n, sub["n"].to_numpy(), sub["build_s"].to_numpy())
            per_feature_s = _loglog_predict(
                n, sub["n"].to_numpy(), sub["per_feature_s"].to_numpy()
            )
            total_s = build_s + M_FEATURES * per_feature_s if supported else np.nan
            rows.append(
                {
                    "impl": spec.name,
                    "target": target,
                    "n": n,
                    "geometry": geometry,
                    "supported": supported,
                    "build_s": build_s if supported else np.nan,
                    "per_feature_s": per_feature_s if supported else np.nan,
                    "peak_mb": peak_mb,
                    "total_s": total_s,
                }
            )
    return pd.DataFrame(rows)


def _draw_white_noise(n: int, features: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, features))


def _draw_gaussian_process(
    covariance: np.ndarray, features: int, seed: int
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))[None, :]
    return factor @ np.random.default_rng(seed).standard_normal(
        (len(covariance), features)
    )


def _torus_covariance(
    coords: np.ndarray,
    domain: tuple[float, float],
    method: str,
    parameters: dict,
) -> np.ndarray:
    distance = compute_torus_distance_matrix(coords, domain)
    if method == "gaussian":
        return np.exp(-(distance**2) / (2 * parameters["bandwidth"] ** 2))
    if method == "matern":
        bandwidth, nu = parameters["bandwidth"], parameters["nu"]
        safe_distance = np.where(distance == 0, 1e-15, distance)
        scaled = np.sqrt(2 * nu) * safe_distance / bandwidth
        covariance = 2 ** (1 - nu) / gamma(nu) * scaled**nu * kv(nu, scaled)
        np.fill_diagonal(covariance, 1)
        return covariance
    raise ValueError(f"Torus covariance is undefined for {method}")


def _build_matched_matrix_fft_kernels(
    ny: int, nx: int, method: str, parameters: dict
) -> tuple[np.ndarray, MatrixKernel, FFTKernel]:
    coords = _regular_grid_coordinates(ny, nx)
    domain = (float(ny), float(nx))
    # Use a torus in both implementations so agreement tests compare algorithms,
    # not boundary conditions.
    if method in {"gaussian", "matern"}:
        matrix = MatrixKernel.from_matrix(
            _torus_covariance(coords, domain, method, parameters), method=method
        )
    else:
        distance = compute_torus_distance_matrix(coords, domain)
        np.fill_diagonal(distance, np.inf)
        k_neighbors = parameters.get("k_neighbors", 4)
        neighbors = np.argpartition(distance, k_neighbors - 1, axis=1)[:, :k_neighbors]
        adjacency = np.zeros_like(distance)
        adjacency[np.arange(len(coords))[:, None], neighbors] = 1
        adjacency = np.maximum(adjacency, adjacency.T)
        degree = adjacency.sum(axis=1)
        normalized = adjacency / np.sqrt(
            np.maximum(degree[:, None] * degree[None, :], 1e-20)
        )
        if method == "car":
            precision = np.eye(len(coords)) - parameters.get("rho", 0.9) * normalized
            matrix = MatrixKernel.from_matrix(
                precision, method=method, is_precision=True
            )
        elif method == "moran":
            matrix = MatrixKernel.from_matrix(normalized, method=method)
        else:
            raise ValueError(f"Unknown validation method: {method}")
    fft = FFTKernel((ny, nx), method=method, workers=1, **parameters)
    return coords, matrix, fft


def _compute_q_scores(kernel, values: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return np.asarray(
        spatial_q_test(
            _reshape_fft_values(kernel, values, ny, nx),
            kernel,
            return_pval=False,
            show_progress=False,
        )
    )


def _compute_pvalues(
    kernel,
    values: np.ndarray,
    ny: int,
    nx: int,
    null_method: str,
) -> np.ndarray:
    null = compute_null_params(kernel, method=null_method)
    _, pvalues = spatial_q_test(
        _reshape_fft_values(kernel, values, ny, nx),
        kernel,
        null_params=null,
        show_progress=False,
    )
    return np.asarray(pvalues)


def _compute_implementation_agreement_tables() -> dict[str, pd.DataFrame]:
    ny = nx = 24
    values = _draw_white_noise(ny * nx, 100, seed=0)
    torus, boundary = [], []
    # Compare identical toroidal operators first, then expose open-boundary error.
    for method, parameters in _VALIDATION_KERNEL_PARAMETERS.items():
        coords, matrix, fft = _build_matched_matrix_fft_kernels(
            ny, nx, method, parameters
        )
        q_matrix = _compute_q_scores(matrix, values, ny, nx)
        q_fft = _compute_q_scores(fft, values, ny, nx)
        q_open = _compute_q_scores(
            MatrixKernel(coords, method=method, **parameters), values, ny, nx
        )
        torus.append(pd.DataFrame({"method": method, "Q_M": q_matrix, "Q_F": q_fft}))
        boundary.append(
            pd.DataFrame(
                {
                    "method": method,
                    "Q_M_open": q_open,
                    "Q_F_torus": q_fft,
                }
            )
        )

    n = 5_000
    domain = np.sqrt(n / 2)
    coords = np.random.default_rng(7).uniform(0, domain, size=(n, 2))
    values = _draw_white_noise(n, 80, seed=3)
    smooth, graph = [], []
    # On irregular coordinates, distance and graph kernels exercise NUFFT directly.
    for method, parameters in _VALIDATION_KERNEL_PARAMETERS.items():
        matrix = MatrixKernel(coords, method=method, **parameters)
        nufft = NUFFTKernel(coords, method=method, workers=1, **parameters)
        table = pd.DataFrame(
            {
                "method": method,
                "Q_M": _compute_q_scores(matrix, values, 1, 1),
                "Q_N": _compute_q_scores(nufft, values, 1, 1),
            }
        )
        (smooth if method in {"gaussian", "matern"} else graph).append(table)

    return {
        "torus_agreement": pd.concat(torus, ignore_index=True),
        "boundary_agreement": pd.concat(boundary, ignore_index=True),
        "smooth_agreement": pd.concat(smooth, ignore_index=True),
        "graph_agreement": pd.concat(graph, ignore_index=True),
    }


def _compute_null_calibration_tables() -> dict[str, pd.DataFrame]:
    ny = nx = 32
    n = ny * nx
    n_reps = 2_000
    alpha = 0.05
    seed = 2024
    psd_parameters = {
        "gaussian": {"bandwidth": 2.5},
        "matern": {"bandwidth": 2.5, "nu": 1.5},
        "car": {"k_neighbors": 4, "rho": 0.9},
    }
    regular_null = _draw_white_noise(n, n_reps, seed)
    regular_pairs = _draw_white_noise(n, n_reps, seed + 11)
    domain = np.sqrt(n / 2)
    irregular_coords = np.random.default_rng(11).uniform(0, domain, size=(n, 2))
    irregular_null = _draw_white_noise(n, n_reps, seed + 100)
    irregular_pairs = _draw_white_noise(n, n_reps, seed + 21)
    fpr_rows: list[dict] = []
    pair_tables = []

    # Reuse each seeded null draw across implementations so discrepancies are paired.

    def add_fpr(grid: str, method: str, impl: str, null: str, pvalues) -> None:
        pvalues = np.asarray(pvalues)
        rate = float((pvalues < alpha).mean())
        error = np.sqrt(max(rate * (1 - rate), 1e-12) / len(pvalues))
        fpr_rows.append(
            {
                "grid": grid,
                "method": method,
                "impl": impl,
                "null": null,
                "fpr": rate,
                "se": error,
            }
        )

    def add_pairs(grid: str, method: str, impl: str, kernel, values) -> None:
        pair_tables.append(
            pd.DataFrame(
                {
                    "grid": grid,
                    "method": method,
                    "impl": impl,
                    "p_welch": _compute_pvalues(kernel, values, ny, nx, "welch"),
                    "p_liu": _compute_pvalues(kernel, values, ny, nx, "liu"),
                }
            )
        )

    for method, parameters in psd_parameters.items():
        # On regular coordinates, compare matrix and FFT implementations.
        _, matrix, fft = _build_matched_matrix_fft_kernels(ny, nx, method, parameters)
        for impl, kernel in (("Matrix", matrix), ("FFT", fft)):
            for null in ("welch", "liu"):
                add_fpr(
                    "regular",
                    method,
                    impl,
                    null,
                    _compute_pvalues(kernel, regular_null, ny, nx, null),
                )
            add_pairs("regular", method, impl, kernel, regular_pairs)

        # On irregular coordinates, compare matrix and NUFFT implementations.
        matrix = MatrixKernel(irregular_coords, method=method, **parameters)
        nufft = NUFFTKernel(irregular_coords, method=method, workers=1, **parameters)
        for impl, kernel in (("Matrix", matrix), ("NUFFT", nufft)):
            for null in ("welch", "liu"):
                add_fpr(
                    "irregular",
                    method,
                    impl,
                    null,
                    _compute_pvalues(kernel, irregular_null, ny, nx, null),
                )
            add_pairs("irregular", method, impl, kernel, irregular_pairs)

    # Compare matrix and FFT implementations of Moran's I.
    _, matrix, fft = _build_matched_matrix_fft_kernels(
        ny, nx, "moran", _VALIDATION_KERNEL_PARAMETERS["moran"]
    )
    moran_regular = _draw_white_noise(n, n_reps, seed + 1)
    for impl, kernel in (("Matrix", matrix), ("FFT", fft)):
        add_fpr(
            "regular",
            "moran",
            impl,
            "clt",
            _compute_pvalues(kernel, moran_regular, ny, nx, "clt"),
        )
    matrix = MatrixKernel(
        irregular_coords,
        method="moran",
        **_VALIDATION_KERNEL_PARAMETERS["moran"],
    )
    nufft = NUFFTKernel(
        irregular_coords,
        method="moran",
        workers=1,
        **_VALIDATION_KERNEL_PARAMETERS["moran"],
    )
    for impl, kernel in (("Matrix", matrix), ("NUFFT", nufft)):
        add_fpr(
            "irregular",
            "moran",
            impl,
            "clt",
            _compute_pvalues(kernel, irregular_null, ny, nx, "clt"),
        )

    return {
        "fpr": pd.DataFrame(fpr_rows),
        "welch_liu": pd.concat(pair_tables, ignore_index=True),
    }


def _compute_implementation_power_table(
    coords: np.ndarray,
    signal: np.ndarray,
    noise: np.ndarray,
    shape: tuple[int, int],
    spectral_impl: str,
) -> pd.DataFrame:
    ny, nx = shape
    rows = []
    for method, parameters in _VALIDATION_KERNEL_PARAMETERS.items():
        if spectral_impl == "FFT":
            _, matrix, spectral = _build_matched_matrix_fft_kernels(
                ny, nx, method, parameters
            )
        elif spectral_impl == "NUFFT":
            matrix = MatrixKernel(coords, method=method, **parameters)
            spectral = NUFFTKernel(coords, method=method, workers=1, **parameters)
        else:
            raise ValueError(f"Unknown spectral implementation: {spectral_impl}")
        nulls = ("clt",) if method == "moran" else ("welch", "liu")
        kernels = (("Matrix", matrix), (spectral_impl, spectral))
        null_parameters = {
            (impl, null): compute_null_params(kernel, method=null)
            for impl, kernel in kernels
            for null in nulls
        }

        # Compute test power for each implementation and null method across SNRs.
        for snr in _POWER_SIGNAL_TO_NOISE_RATIOS:
            values = snr * signal + noise
            for impl, kernel in kernels:
                for null in nulls:
                    _, pvalues = spatial_q_test(
                        _reshape_fft_values(kernel, values, ny, nx),
                        kernel,
                        null_params=null_parameters[(impl, null)],
                        show_progress=False,
                    )
                    rows.append(
                        {
                            "method": method,
                            "impl": impl,
                            "null": null,
                            "snr": snr,
                            "power": float((np.asarray(pvalues) < 0.05).mean()),
                        }
                    )
    return pd.DataFrame(rows)


def _compute_hyperparameter_power_table(
    coords: np.ndarray, signal: np.ndarray, noise: np.ndarray
) -> pd.DataFrame:
    sweeps = {
        "gaussian": ("bandwidth", (0.5, 1.5, 3.0, 5.0, 8.0)),
        "matern": ("bandwidth", (0.5, 1.5, 3.0, 5.0, 8.0)),
        "car": ("rho", (0.3, 0.5, 0.7, 0.9, 0.99)),
        "moran": ("k_neighbors", (4, 8, 16, 32, 64)),
    }
    preferred_null = {
        "gaussian": "welch",
        "matern": "welch",
        "car": "welch",
        "moran": "clt",
    }
    rows = []
    for method, (parameter, values_to_test) in sweeps.items():
        for value in values_to_test:
            parameters = dict(_VALIDATION_KERNEL_PARAMETERS[method])
            parameters[parameter] = value
            kernel = MatrixKernel(coords, method=method, **parameters)
            null = preferred_null[method]
            null_parameters = compute_null_params(kernel, method=null)
            for snr in _POWER_SIGNAL_TO_NOISE_RATIOS:
                _, pvalues = spatial_q_test(
                    snr * signal + noise,
                    kernel,
                    null_params=null_parameters,
                    show_progress=False,
                )
                rows.append(
                    {
                        "method": method,
                        "hp_name": parameter,
                        "hp_value": value,
                        "snr": snr,
                        "power": float((np.asarray(pvalues) < 0.05).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _compute_power_comparison_tables() -> dict[str, pd.DataFrame]:
    ny = nx = 28
    n = ny * nx
    n_reps = 200
    coords = _regular_grid_coordinates(ny, nx)
    regular_covariance = _torus_covariance(
        coords,
        (float(ny), float(nx)),
        "matern",
        {"bandwidth": 3.0, "nu": 1.5},
    )
    # Matérn GP draws provide the same known alternative to both implementations.
    regular_signal = _draw_gaussian_process(regular_covariance, n_reps, seed=99)
    regular_noise = _draw_white_noise(n, n_reps, seed=100)
    domain = np.sqrt(n / 2)
    irregular_coords = np.random.default_rng(51).uniform(0, domain, size=(n, 2))
    irregular_covariance = MatrixKernel(
        irregular_coords, method="matern", bandwidth=3.0, nu=1.5
    ).realization()
    irregular_signal = _draw_gaussian_process(irregular_covariance, n_reps, seed=199)
    irregular_noise = _draw_white_noise(n, n_reps, seed=200)

    return {
        "power_regular": _compute_implementation_power_table(
            coords, regular_signal, regular_noise, (ny, nx), "FFT"
        ),
        "power_irregular": _compute_implementation_power_table(
            irregular_coords,
            irregular_signal,
            irregular_noise,
            (ny, nx),
            "NUFFT",
        ),
        "power_hyperparameter": _compute_hyperparameter_power_table(
            coords, regular_signal, regular_noise
        ),
    }


def cache_validation_benchmark_results(
    *, force: bool = False
) -> dict[str, pd.DataFrame]:
    """Cache implementation agreement, null calibration, and power tables."""
    ensure_output_dirs()
    VALIDATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stages = (
        (
            "implementation agreement",
            (
                "torus_agreement",
                "boundary_agreement",
                "smooth_agreement",
                "graph_agreement",
            ),
            _compute_implementation_agreement_tables,
        ),
        ("null calibration", ("fpr", "welch_liu"), _compute_null_calibration_tables),
        (
            "power comparison",
            ("power_regular", "power_irregular", "power_hyperparameter"),
            _compute_power_comparison_tables,
        ),
    )
    for label, keys, compute in stages:
        paths = [VALIDATION_RESULT_DIR / VALIDATION_CACHE_FILES[key] for key in keys]
        # Recompute a whole stage if any member is absent, keeping related tables
        # on the same simulation seed and parameter set.
        if cache_is_reusable(
            *paths,
            force=force,
            inputs=CACHE_INPUTS,
            logger=LOGGER,
            label=f"Validation {label}",
        ):
            continue
        tables = compute()
        if set(tables) != set(keys):
            raise RuntimeError(
                f"Validation stage returned {sorted(tables)}, expected {keys}"
            )
        for key, table in tables.items():
            table.to_csv(
                VALIDATION_RESULT_DIR / VALIDATION_CACHE_FILES[key], index=False
            )
        LOGGER.info("Validation %s: cached %d tables", label, len(tables))
    return load_validation_benchmark_results()


def load_validation_benchmark_results() -> dict[str, pd.DataFrame]:
    """Load the seeded implementation, calibration, and power caches."""
    results = {}
    for key, name in VALIDATION_CACHE_FILES.items():
        path = VALIDATION_RESULT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing validation cache: {path}")
        results[key] = pd.read_csv(path)
    return results


def _matern_field(
    shape: tuple[int, int], *, spacing: float = 1.0, bandwidth: float = 3.0, seed: int
) -> np.ndarray:
    """Draw a periodic Matérn-nu=1.5 field by spectral filtering."""
    fy = 2 * np.pi * scipy.fft.fftfreq(shape[0], d=spacing)
    fx = 2 * np.pi * scipy.fft.fftfreq(shape[1], d=spacing)
    density = (3.0 / bandwidth**2 + fy[:, None] ** 2 + fx[None, :] ** 2) ** -2.5
    noise = np.random.default_rng(seed).standard_normal(shape)
    field = scipy.fft.ifft2(scipy.fft.fft2(noise) * np.sqrt(density)).real
    return (field - field.mean()) / field.std()


def simulate_validation_signal_examples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic regular- and irregular-grid Matérn signal examples."""
    regular = _matern_field((32, 32), seed=11)
    n = 5_000
    domain = np.sqrt(n / 2)
    coords = np.random.default_rng(7).uniform(0, domain, size=(n, 2))
    fine = _matern_field((128, 128), spacing=domain / 128, seed=13)
    values = map_coordinates(
        fine,
        np.vstack((coords[:, 0], coords[:, 1])) * (128 / domain),
        order=1,
        mode="wrap",
    )
    return regular, coords, values
