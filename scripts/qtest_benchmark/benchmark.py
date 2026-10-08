"""Runtime, implementation-agreement, calibration, and power benchmarks."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import scipy.fft
from joblib import Parallel, delayed
from scipy.fft import prev_fast_len
from scipy.ndimage import map_coordinates
from scipy.special import gamma, kv
from sonic.kernels import MatrixKernel
from sonic.kernels.fft import FFTKernel
from sonic.kernels.nufft import NUFFTKernel
from sonic.statistics import _q_pvalues, compute_null_params, spatial_q_test
from sonic.utils import (
    _chunk_parameters,
    auto_chunk_size,
    compute_torus_distance_matrix,
    resolve_parallelism,
)
from threadpoolctl import threadpool_limits

from scripts.paper_utils import (
    PROJECT_ROOT,
    cache_entry_is_current,
    cache_signature,
    max_rss_gb,
    source_provenance,
    update_cache_entry,
)

from .config import (
    BENCHMARK_PROJECTION_FILE,
    BENCHMARK_RUNS_FILE,
    BENCHMARK_SCALING_FILE,
    BENCHMARK_TARGETS,
    M_FEATURES,
    RESULT_DIR,
    RUNTIME_CORES,
    RUNTIME_JITTER_FRACTION,
    RUNTIME_MEMORY_BUDGET_BYTES,
    RUNTIME_REPEATS,
    VALIDATION_RESULT_DIR,
    ensure_output_dirs,
)
from .utils import cache_is_reusable

LOGGER = logging.getLogger(__name__)

# Fixed measurement protocol; experiment settings live in config.py.
_RUNTIME_HEADROOM_BYTES = 2 * 1024**3
_RUNTIME_MEASURE_FEATURES = 256
_RUNTIME_TIMEOUT_S = 600
_RUNTIME_COORDINATE_SEED = 42

VALIDATION_CACHE_FILES = {
    "torus_agreement": "agreement_torus.csv",
    "boundary_agreement": "agreement_open_boundary.csv",
    "smooth_agreement": "agreement_nufft_distance.csv",
    "graph_agreement": "agreement_nufft_graph.csv",
    "fpr": "null_fpr.csv",
    "welch_moments": "null_welch_moments.csv",
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
    RuntimeImplementation("Matrix-CAR (sparse)", 3_000_000, "welch", "matrix_car"),
    RuntimeImplementation("Matrix-Moran (sparse)", 6_000_000, "clt", "matrix_moran"),
    RuntimeImplementation("FFT-Matern", 10_000_000, "moments", "fft"),
    RuntimeImplementation("NUFFT-Matern", 6_000_000, "moments", "nufft"),
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

VALIDATION_KERNEL_PARAMETERS = {
    "gaussian": {"bandwidth": 2.0},
    "matern": {"bandwidth": 2.0, "nu": 1.5},
    "car": {"k_neighbors": 4, "rho": 0.9},
    "moran": {"k_neighbors": 4},
}
POWER_SIGNAL_TO_NOISE_RATIOS = (
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
    # Preserve target sample counts when a reasonably shaped rectangle exists.
    for side in range(int(np.sqrt(n)), 3, -1):
        if n % side == 0:
            return side, n // side
    side = prev_fast_len(max(int(np.sqrt(n)), 8))
    return side, side


def _regular_grid_coordinates(ny: int, nx: int) -> np.ndarray:
    y, x = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    return np.column_stack((y.ravel(), x.ravel()))


def _build_runtime_kernel(spec, ny, nx, coords, *, workers=1, geometry="regular"):
    if spec.kind == "matrix_matern":
        return MatrixKernel(coords, method="matern", bandwidth=2.5, nu=1.5)
    if spec.kind == "matrix_car":
        return MatrixKernel(coords, method="car", k_neighbors=4, rho=0.9)
    if spec.kind == "matrix_moran":
        return MatrixKernel(coords, method="moran", k_neighbors=4)
    if spec.kind == "fft":
        return FFTKernel((ny, nx), method="matern", bandwidth=2.5, nu=1.5, workers=workers)
    if spec.kind == "nufft":
        grid = {"grid_shape": (ny, nx), "spacing": (1.0, 1.0)} if geometry == "regular" else {}
        return NUFFTKernel(
            coords,
            method="matern",
            bandwidth=2.5,
            nu=1.5,
            workers=workers,
            nthreads=workers,
            **grid,
        )
    raise ValueError(f"Unknown implementation kind: {spec.kind}")


def _reshape_fft_values(kernel, values: np.ndarray, ny: int, nx: int) -> np.ndarray:
    return values.reshape(ny, nx, -1) if isinstance(kernel, FFTKernel) else values


def runtime_provenance() -> dict:
    """Fingerprint numerical dependencies and actual imported SONIC source."""
    return {
        **source_provenance(__file__),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _runtime_parallelism(case: dict, **kwargs) -> tuple[int, int]:
    """Use SONIC's policy with the declared CPU allowance, without timing a sweep."""
    cores = min(case["cores"], os.cpu_count() or 1)
    backend = case["kind"] if case["kind"] in {"fft", "nufft"} else "matrix"
    # The scheduler reads os.cpu_count; expose the benchmark allowance only
    # while resolving it. Actual computation uses the returned explicit counts.
    with patch("sonic.utils.os.cpu_count", return_value=cores):
        return resolve_parallelism(case["jobs"], case["workers"], backend=backend, **kwargs)


def _runtime_case(case: dict) -> dict:  # noqa: C901 — explicit benchmark phases and failure states
    """Measure setup and Q-tests for one case in a fresh worker process.

    Scaling cases stop after kernel construction, null calibration, and one
    feature's Q-test. Production cases also measure throughput in bounded
    batches; the caller extrapolates that rate to the full gene count.

    The case's memory limit covers total RSS, including the coordinator.
    SONIC receives only the workspace allowance left after retained state,
    input batches, and safety headroom. The parent monitors actual peak RSS.
    """
    # Resolve the initial CPU split for construction and null calibration.
    spec = next(item for item in RUNTIME_IMPLEMENTATIONS if item.kind == case["kind"])
    n_jobs, n_workers = _runtime_parallelism(case)
    blas_threads = case["blas_threads"]
    if (
        min(n_jobs, n_workers, blas_threads) < 1
        or n_jobs * max(n_workers, blas_threads) > case["cores"]
    ):
        raise ValueError("Job/thread split exceeds the declared CPU budget")

    # Generate geometry and input outside the timed sections. Jitter retains
    # local grid structure while giving NUFFT irregular coordinates.
    ny, nx = _bounded_fft_grid_shape(case["requested_n"])
    rng = np.random.default_rng(_RUNTIME_COORDINATE_SEED)
    if case["geometry"] == "irregular":
        n = case["requested_n"]
        # Preserve n even when the regular FFT shape rounded down.
        nx = (n + ny - 1) // ny
        coords = _regular_grid_coordinates(ny, nx)[:n]
        coords += rng.uniform(-RUNTIME_JITTER_FRACTION, RUNTIME_JITTER_FRACTION, coords.shape)
    else:
        n = ny * nx
        coords = _regular_grid_coordinates(ny, nx)
    first_feature = rng.standard_normal((n, 1))
    total_budget_bytes = case["total_memory_bytes"]
    coordinator_rss_bytes = case["coordinator_rss_bytes"]
    process = psutil.Process()
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        # Time construction and calibration separately; reuse this null fit
        # for every subsequent Q-test so setup is charged only once.
        started = time.perf_counter()
        kernel = _build_runtime_kernel(
            spec, ny, nx, coords, workers=n_workers, geometry=case["geometry"]
        )
        kernel_s = time.perf_counter() - started
        started = time.perf_counter()
        null_params = compute_null_params(kernel)
        calibration_s = time.perf_counter() - started
        if null_params["method"] != spec.null_method:
            raise ValueError(
                f"Default calibration changed: {null_params['method']} != {spec.null_method}"
            )

        # The first feature is already resident, so subtract current RSS and
        # headroom before assigning its temporary workspace allowance.
        first_q_workspace_bytes = (
            total_budget_bytes
            - coordinator_rss_bytes
            - process.memory_info().rss
            - _RUNTIME_HEADROOM_BYTES
        )
        if first_q_workspace_bytes <= 0:
            raise MemoryError("Retained state leaves no safe workspace")
        started = time.perf_counter()
        spatial_q_test(
            _reshape_fft_values(kernel, first_feature, ny, nx),
            kernel,
            null_params=null_params,
            memory_budget_bytes=first_q_workspace_bytes,
        )
        first_q_s = time.perf_counter() - started
        del first_feature
        result = {
            "jobs": n_jobs,
            "workers": n_workers,
            "calibration_workers": n_workers,
            "coordinate_model": "jittered_grid"
            if case["geometry"] == "irregular"
            else "regular_grid",
            "jitter_fraction": RUNTIME_JITTER_FRACTION if case["geometry"] == "irregular" else 0.0,
            "coordinate_seed": _RUNTIME_COORDINATE_SEED,
            "n": n,
            "grid_ny": getattr(kernel, "grid_shape", (ny, nx))[0],
            "grid_nx": getattr(kernel, "grid_shape", (ny, nx))[1],
            "spacing": list(getattr(kernel, "spacing", (1.0, 1.0))),
            "kernel_s": kernel_s,
            "calibration_s": calibration_s,
            "build_s": kernel_s + calibration_s,
            "first_q_s": first_q_s,
            "single_call_s": kernel_s + calibration_s + first_q_s,
            "null_method": null_params["method"],
            "null_source": null_params.get("source", "traces"),
            "null_fit": null_params.get("q_fit", {}).get("fit", null_params["method"]),
            "stores_precision": bool(getattr(kernel, "stores_precision", False)),
            "input_layout": "float64, observation-by-feature, Fortran contiguous",
        }
        # Single-core scaling ends here; production continues with batches.
        if case["phase"] == "scaling":
            result["peak_rss_bytes"] = int(max_rss_gb() * 1024**3) + coordinator_rss_bytes
            return result

        # Production memory accounting: total RSS budget = retained state +
        # headroom + input batches + SONIC workspace. A wave holds one batch
        # per outer job; transform threads within a job share its batch.
        retained_rss_bytes = process.memory_info().rss + coordinator_rss_bytes
        batch_budget_bytes = total_budget_bytes - retained_rss_bytes - _RUNTIME_HEADROOM_BYTES
        if batch_budget_bytes <= 0:
            raise MemoryError("Retained state leaves no safe feature-wave workspace")
        # Split the remaining allowance in proportion to per-feature costs.
        # Both inputs and workspace grow with n_jobs * chunk_size.
        _, workspace_bytes_per_feature = _chunk_parameters(kernel)
        input_bytes_per_feature = n * np.dtype(float).itemsize
        workspace_budget_bytes = (
            batch_budget_bytes
            * workspace_bytes_per_feature
            // (workspace_bytes_per_feature + input_bytes_per_feature)
        )
        try:
            # Estimate task count from a single-job chunk, then resolve jobs
            # under the CPU and memory limits before sizing concurrent chunks.
            single_job_chunk_size = auto_chunk_size(kernel, budget_bytes=workspace_budget_bytes)
            n_jobs, n_workers = _runtime_parallelism(
                case,
                n_tasks=max(1, (M_FEATURES + single_job_chunk_size - 1) // single_job_chunk_size),
                per_feat_bytes=workspace_bytes_per_feature,
                budget_bytes=workspace_budget_bytes,
            )
            chunk_size = auto_chunk_size(kernel, n_jobs=n_jobs, budget_bytes=workspace_budget_bytes)
        except ValueError as exc:
            raise MemoryError(str(exc)) from exc
        # Construction used the initial split; batched tests use this final
        # split, including FINUFFT's separate native thread setting.
        kernel.workers = n_workers
        if spec.kind == "nufft":
            kernel.nthreads = n_workers
        input_wave_bytes = n * chunk_size * n_jobs * np.dtype(float).itemsize
        result.update(
            jobs=n_jobs,
            workers=n_workers,
            chunk_size=chunk_size,
            workspace_budget_bytes=workspace_budget_bytes,
            retained_rss_bytes=retained_rss_bytes,
            input_wave_bytes=input_wave_bytes,
        )

        def test_block(block):
            return spatial_q_test(
                _reshape_fft_values(kernel, block, ny, nx),
                kernel,
                null_params=null_params,
                chunk_size=chunk_size,
                memory_budget_bytes=workspace_budget_bytes // n_jobs,
            )

        def run_wave(blocks, parallel):
            return parallel(delayed(test_block)(block) for block in blocks)

        with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
            # Exclude thread-pool startup and first-use effects from throughput.
            warmup = rng.standard_normal((n, min(4, chunk_size)))
            run_wave([warmup] * n_jobs, parallel)
            del warmup
            # Measure at least three full waves, with a minimum feature count.
            features_to_measure = max(_RUNTIME_MEASURE_FEATURES, 3 * n_jobs * chunk_size)
            rng = np.random.default_rng(20261007)
            measured_features = batch_test_s = 0
            while measured_features < features_to_measure:
                # Allocate at most one batch per job. Generation and validation
                # are excluded from compute time but still count toward RSS.
                blocks = []
                wave_features = 0
                for _ in range(n_jobs):
                    block_features = min(
                        chunk_size, features_to_measure - measured_features - wave_features
                    )
                    if block_features <= 0:
                        break
                    block = np.empty((n, block_features), dtype=float, order="F")
                    rng.standard_normal(out=block.ravel(order="F"))
                    blocks.append(block)
                    wave_features += block_features
                # Time only parallel dispatch and Q/p-value computation.
                started = time.perf_counter()
                wave_results = run_wave(blocks, parallel)
                batch_test_s += time.perf_counter() - started
                for q, p in wave_results:
                    if (
                        not np.all(np.isfinite(q))
                        or not np.all(np.isfinite(p))
                        or np.any((p < 0) | (p > 1))
                    ):
                        raise ValueError("Non-finite Q or invalid p-value")
                measured_features += wave_features
                # Release the current wave before allocating the next one.
                del block, blocks, wave_results
        result.update(
            measured_features=measured_features,
            batch_test_s=batch_test_s,
            features_per_second=measured_features / batch_test_s,
            per_feature_s=batch_test_s / measured_features,
            peak_rss_bytes=int(max_rss_gb() * 1024**3) + coordinator_rss_bytes,
        )
        return result


def _write_runtime_result(request: str, response: str) -> None:
    """Private subprocess target; workflow.py owns the command-line interface."""
    try:
        payload = {**_runtime_case(json.loads(Path(request).read_text())), "status": "ok"}
    except MemoryError as exc:
        payload = {"status": "memory_limit", "error": str(exc)}
    except Exception as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    Path(response).write_text(json.dumps(payload, allow_nan=False))


def _run_isolated_case(case: dict) -> dict:  # noqa: C901 — explicit benchmark phases and failure states
    """Run one case in a fresh interpreter and return its measurements/status.

    The RSS limit covers the coordinator, worker, and worker descendants.
    The timeout includes interpreter startup and all measurement phases;
    compute timings come separately from the worker's result.

    Resource-limit and worker failures remain explicit so the caller can
    exclude failed cases from runtime projections.
    """
    # Copy the request so adding coordinator memory does not modify the caller's case.
    case = dict(case, coordinator_rss_bytes=psutil.Process().memory_info().rss)

    # Set native thread defaults before importing numerical libraries in the
    # worker; SONIC sets FFT/NUFFT thread counts explicitly for each case.
    worker_env = dict(os.environ)
    for env_key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        worker_env[env_key] = str(case["blas_threads"])
    worker_env.update(
        OMP_NUM_THREADS=str(min(4, case["cores"])),
        NUMEXPR_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )

    sampled_peak_rss_bytes = 0
    status, error = "ok", ""
    # Keep structured results separate from library output and diagnostics.
    # All per-case files are removed when this context exits.
    with tempfile.TemporaryDirectory(prefix="sonic-qtest-") as case_dir:
        case_dir = Path(case_dir)
        request_path, response_path = case_dir / "case.json", case_dir / "result.json"
        request_path.write_text(json.dumps(case))
        with (case_dir / "worker.log").open("w+") as worker_log:
            # Start a fresh interpreter so cases do not inherit warmed kernels,
            # cached factorizations, or memory retained by earlier measurements.
            case_started_at = time.monotonic()
            with subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    "import sys\n"
                    "from scripts.qtest_benchmark.benchmark import _write_runtime_result\n"
                    "_write_runtime_result(*sys.argv[1:])",
                    str(request_path),
                    str(response_path),
                ],
                cwd=PROJECT_ROOT,
                env=worker_env,
                stdout=worker_log,
                stderr=worker_log,
            ) as worker:
                worker_process = psutil.Process(worker.pid)
                try:
                    while worker.poll() is None:
                        # Poll total resident memory, including native allocations.
                        # Processes can exit between enumeration and RSS reads.
                        try:
                            total_rss_bytes = psutil.Process().memory_info().rss
                            for tree_process in [
                                worker_process,
                                *worker_process.children(recursive=True),
                            ]:
                                try:
                                    total_rss_bytes += tree_process.memory_info().rss
                                except psutil.NoSuchProcess:
                                    pass
                            sampled_peak_rss_bytes = max(sampled_peak_rss_bytes, total_rss_bytes)
                        except psutil.NoSuchProcess:
                            pass
                        # These are monitored limits, not OS allocation caps:
                        # detect an overrun, stop polling, and clean up below.
                        if sampled_peak_rss_bytes > case["total_memory_bytes"]:
                            status, error = (
                                "memory_limit",
                                "Monitored process-tree RSS exceeded budget",
                            )
                            break
                        if time.monotonic() - case_started_at > case.get(
                            "timeout_s", _RUNTIME_TIMEOUT_S
                        ):
                            status, error = "timeout", "Case exceeded wall-clock timeout"
                            break
                        time.sleep(0.02)
                except (PermissionError, psutil.AccessDenied) as exc:
                    # Do not accept an unmonitored case as a valid measurement.
                    status, error = "monitor_error", f"Cannot monitor process-tree RSS: {exc}"
                finally:
                    # If monitoring stops early, kill reachable descendants first,
                    # then kill and reap the worker before removing its files.
                    if worker.poll() is None:
                        try:
                            descendants = worker_process.children(recursive=True)
                        except (PermissionError, psutil.AccessDenied, psutil.NoSuchProcess):
                            descendants = []
                        for tree_process in descendants:
                            try:
                                tree_process.kill()
                            except psutil.NoSuchProcess:
                                pass
                        worker.kill()
                        worker.wait()
            # Preserve monitoring failures. A normal process exit may still
            # carry a numerical/memory failure reported in the result JSON.
            worker_result = (
                json.loads(response_path.read_text())
                if status == "ok" and worker.returncode == 0 and response_path.exists()
                else {}
            )
            if status == "ok":
                status = worker_result.get("status", "error")
                error = worker_result.get("error", "")
                if worker.returncode != 0 or not worker_result:
                    # Crashes or missing results need the captured log for diagnosis.
                    status = "error"
                    worker_log.seek(0)
                    error = f"Worker exit {worker.returncode}: {worker_log.read()[-2000:]}"
    # Keep the larger of sampled process-tree RSS and the worker's reported
    # high-water mark, which can catch brief peaks between polling intervals.
    result = {**case, **worker_result, "status": status, "error": error}
    result["peak_rss_bytes"] = max(sampled_peak_rss_bytes, result.get("peak_rss_bytes", 0))
    # A worker-reported peak can reveal an overrun even after a clean exit.
    if result["peak_rss_bytes"] > case["total_memory_bytes"]:
        result.update(status="memory_limit", error="Peak RSS exceeded budget")
    result["peak_rss_gib"] = result["peak_rss_bytes"] / 1024**3
    return result


def _summarize_runtime(rows: list[dict]) -> dict:
    """Require every independent repetition to succeed; keep dispersion and peak RSS."""
    result = dict(rows[0])
    failed = next((row for row in rows if row["status"] != "ok"), None)
    if failed:
        result.update(status=failed["status"], error=failed["error"])
    for key in (
        "kernel_s",
        "calibration_s",
        "build_s",
        "first_q_s",
        "single_call_s",
        "features_per_second",
        "per_feature_s",
        "batch_test_s",
    ):
        values = [row[key] for row in rows if key in row and row["status"] == "ok"]
        result[key] = float(np.median(values)) if values and not failed else np.nan
        result[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    result["peak_rss_bytes"] = max(row.get("peak_rss_bytes", 0) for row in rows)
    result["peak_rss_gib"] = result["peak_rss_bytes"] / 1024**3
    result.pop("repeat", None)
    result["repetitions"] = len(rows)
    return result


def cache_runtime_scaling_results(*, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: C901 — explicit benchmark phases and failure states
    """Single-core scaling and measured-throughput projections under 8 cores / 16 GiB."""
    ensure_output_dirs()
    # Source provenance includes config.py and the fixed measurement protocol.
    provenance = runtime_provenance()
    signature_data = {
        "provenance": provenance,
        "sample_sizes": BENCHMARK_SAMPLE_SIZES,
        "targets": BENCHMARK_TARGETS,
        "implementations": [vars(spec) for spec in RUNTIME_IMPLEMENTATIONS],
    }
    signature = cache_signature(signature_data)
    outputs = (BENCHMARK_SCALING_FILE, BENCHMARK_PROJECTION_FILE, BENCHMARK_RUNS_FILE)
    if not force and cache_entry_is_current(RESULT_DIR, "runtime", signature, outputs=outputs):
        return pd.read_csv(BENCHMARK_SCALING_FILE), pd.read_csv(BENCHMARK_PROJECTION_FILE)
    if (os.cpu_count() or 1) < RUNTIME_CORES:
        raise RuntimeError("The 8-core projection requires a host exposing at least eight CPUs")
    scaling, projections = [], []
    # An interrupted --force run must not reuse old CSVs with a partial new run log.
    BENCHMARK_PROJECTION_FILE.unlink(missing_ok=True)
    # Append each finished case immediately, so interrupted experiments remain inspectable.
    with BENCHMARK_RUNS_FILE.open("w") as log:

        def measure(case):
            row = _run_isolated_case(case)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if row["status"] == "monitor_error":
                raise RuntimeError(row["error"])
            LOGGER.info(
                "%s n=%s %s %sx%s: %s",
                row["kind"],
                row["requested_n"],
                row["phase"],
                row["jobs"],
                row["workers"],
                row["status"],
            )
            return row

        for spec in RUNTIME_IMPLEMENTATIONS:
            base = {
                "impl": spec.name,
                "kind": spec.kind,
                "total_memory_bytes": RUNTIME_MEMORY_BUDGET_BYTES,
                "cores": RUNTIME_CORES,
                "timeout_s": _RUNTIME_TIMEOUT_S,
            }
            for requested_n in BENCHMARK_SAMPLE_SIZES:
                if requested_n > spec.max_n:
                    continue
                case = dict(
                    base,
                    requested_n=requested_n,
                    geometry="regular",
                    phase="scaling",
                    cores=1,
                    jobs=1,
                    workers=1,
                    blas_threads=1,
                )
                rows = [measure(dict(case, repeat=i)) for i in range(RUNTIME_REPEATS)]
                summary = _summarize_runtime(rows)
                summary.setdefault("n", np.prod(_bounded_fft_grid_shape(requested_n)))
                scaling.append(summary)
                if summary["status"] in {"memory_limit", "timeout"}:
                    break
            for target, n, geometry in BENCHMARK_TARGETS:
                summary = dict(
                    base,
                    target=target,
                    n=n,
                    requested_n=n,
                    geometry=geometry,
                    supported=False,
                    total_s=np.nan,
                    projection_method="target_throughput",
                )
                if geometry == "irregular" and spec.kind == "fft":
                    summary["status"] = "unsupported_geometry"
                elif n > spec.max_n:
                    summary["status"] = "sample_cap"
                else:
                    case = dict(
                        base,
                        requested_n=n,
                        target=target,
                        geometry=geometry,
                        phase="production",
                        jobs="auto",
                        workers="auto",
                        blas_threads=1,
                    )
                    trials = [measure(dict(case, repeat=i)) for i in range(RUNTIME_REPEATS)]
                    measured = _summarize_runtime(trials)
                    summary.update(
                        measured,
                        n=n,
                        measured_n=measured.get("n"),
                        supported=measured["status"] == "ok",
                    )
                    if summary["supported"]:
                        summary["total_s"] = (
                            summary["build_s"] + M_FEATURES * summary["per_feature_s"]
                        )
                projections.append(summary)
    benchmark, projection = pd.DataFrame(scaling), pd.DataFrame(projections)
    benchmark.to_csv(BENCHMARK_SCALING_FILE, index=False)
    projection.to_csv(BENCHMARK_PROJECTION_FILE, index=False)
    update_cache_entry(RESULT_DIR, "runtime", signature, outputs, signature_data=signature_data)
    return benchmark, projection


def _draw_white_noise(n: int, features: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, features))


def _draw_gaussian_process(covariance: np.ndarray, features: int, seed: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))[None, :]
    return factor @ np.random.default_rng(seed).standard_normal((len(covariance), features))


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
        normalized = adjacency / np.sqrt(np.maximum(degree[:, None] * degree[None, :], 1e-20))
        if method == "car":
            precision = np.eye(len(coords)) - parameters.get("rho", 0.9) * normalized
            matrix = MatrixKernel.from_matrix(precision, method=method, is_precision=True)
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
    null_parameters: dict[str, dict],
) -> dict[str, np.ndarray]:
    """Score once, then apply each prepared calibration to the same Q values."""
    scores = _compute_q_scores(kernel, values, ny, nx)
    valid = np.var(np.asarray(values, dtype=float), axis=0, ddof=1) > 1e-24
    return {
        name: np.where(valid, _q_pvalues(scores, params), 1.0)
        for name, params in null_parameters.items()
    }


def _compute_implementation_agreement_tables() -> dict[str, pd.DataFrame]:
    ny = nx = 24
    values = _draw_white_noise(ny * nx, 100, seed=0)
    torus, boundary = [], []
    # Compare identical toroidal operators first, then expose open-boundary error.
    for method, parameters in VALIDATION_KERNEL_PARAMETERS.items():
        coords, matrix, fft = _build_matched_matrix_fft_kernels(ny, nx, method, parameters)
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
    for method, parameters in VALIDATION_KERNEL_PARAMETERS.items():
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

    def add_pairs(grid: str, method: str, impl: str, kernel, values, null_parameters) -> None:
        pvalues = _compute_pvalues(kernel, values, ny, nx, null_parameters)
        pair_tables.append(
            pd.DataFrame(
                {
                    "grid": grid,
                    "method": method,
                    "impl": impl,
                    "p_welch": pvalues["welch"],
                    "p_moments": pvalues["moments"],
                }
            )
        )

    for method, parameters in psd_parameters.items():
        # On regular coordinates, compare matrix and FFT implementations.
        _, matrix, fft = _build_matched_matrix_fft_kernels(ny, nx, method, parameters)
        for impl, kernel in (("Matrix", matrix), ("FFT", fft)):
            null_parameters = {
                null: compute_null_params(kernel, method=null) for null in ("welch", "moments")
            }
            pvalues = _compute_pvalues(kernel, regular_null, ny, nx, null_parameters)
            for null, values in pvalues.items():
                add_fpr(
                    "regular",
                    method,
                    impl,
                    null,
                    values,
                )
            add_pairs("regular", method, impl, kernel, regular_pairs, null_parameters)

        # On irregular coordinates, compare matrix and NUFFT implementations.
        matrix = MatrixKernel(irregular_coords, method=method, **parameters)
        nufft = NUFFTKernel(irregular_coords, method=method, workers=1, **parameters)
        for impl, kernel in (("Matrix", matrix), ("NUFFT", nufft)):
            null_parameters = {
                null: compute_null_params(kernel, method=null) for null in ("welch", "moments")
            }
            pvalues = _compute_pvalues(kernel, irregular_null, ny, nx, null_parameters)
            for null, values in pvalues.items():
                add_fpr(
                    "irregular",
                    method,
                    impl,
                    null,
                    values,
                )
            add_pairs("irregular", method, impl, kernel, irregular_pairs, null_parameters)

    # Compare matrix and FFT implementations of Moran's I.
    _, matrix, fft = _build_matched_matrix_fft_kernels(
        ny, nx, "moran", VALIDATION_KERNEL_PARAMETERS["moran"]
    )
    moran_regular = _draw_white_noise(n, n_reps, seed + 1)
    for impl, kernel in (("Matrix", matrix), ("FFT", fft)):
        add_fpr(
            "regular",
            "moran",
            impl,
            "clt",
            _compute_pvalues(
                kernel,
                moran_regular,
                ny,
                nx,
                {"clt": compute_null_params(kernel, method="clt")},
            )["clt"],
        )
    matrix = MatrixKernel(
        irregular_coords,
        method="moran",
        **VALIDATION_KERNEL_PARAMETERS["moran"],
    )
    nufft = NUFFTKernel(
        irregular_coords,
        method="moran",
        workers=1,
        **VALIDATION_KERNEL_PARAMETERS["moran"],
    )
    for impl, kernel in (("Matrix", matrix), ("NUFFT", nufft)):
        add_fpr(
            "irregular",
            "moran",
            impl,
            "clt",
            _compute_pvalues(
                kernel,
                irregular_null,
                ny,
                nx,
                {"clt": compute_null_params(kernel, method="clt")},
            )["clt"],
        )

    return {
        "fpr": pd.DataFrame(fpr_rows),
        "welch_moments": pd.concat(pair_tables, ignore_index=True),
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
    for method, parameters in VALIDATION_KERNEL_PARAMETERS.items():
        if spectral_impl == "FFT":
            _, matrix, spectral = _build_matched_matrix_fft_kernels(ny, nx, method, parameters)
        elif spectral_impl == "NUFFT":
            matrix = MatrixKernel(coords, method=method, **parameters)
            spectral = NUFFTKernel(coords, method=method, workers=1, **parameters)
        else:
            raise ValueError(f"Unknown spectral implementation: {spectral_impl}")
        nulls = ("clt",) if method == "moran" else ("welch", "moments")
        kernels = (("Matrix", matrix), (spectral_impl, spectral))
        null_parameters = {
            (impl, null): compute_null_params(kernel, method=null)
            for impl, kernel in kernels
            for null in nulls
        }

        # Compute test power for each implementation and null method across SNRs.
        for snr in POWER_SIGNAL_TO_NOISE_RATIOS:
            values = snr * signal + noise
            for impl, kernel in kernels:
                calibrated = _compute_pvalues(
                    kernel,
                    values,
                    ny,
                    nx,
                    {null: null_parameters[(impl, null)] for null in nulls},
                )
                for null, pvalues in calibrated.items():
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
            parameters = dict(VALIDATION_KERNEL_PARAMETERS[method])
            parameters[parameter] = value
            kernel = MatrixKernel(coords, method=method, **parameters)
            null = preferred_null[method]
            null_parameters = compute_null_params(kernel, method=null)
            for snr in POWER_SIGNAL_TO_NOISE_RATIOS:
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


def cache_validation_benchmark_results(*, force: bool = False) -> dict[str, pd.DataFrame]:
    """Cache implementation agreement, null calibration, and power tables."""
    ensure_output_dirs()
    VALIDATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = tuple(VALIDATION_RESULT_DIR / name for name in VALIDATION_CACHE_FILES.values())
    signature_data = {"source": source_provenance(__file__)}
    signature = cache_signature(signature_data)
    force = force or not cache_entry_is_current(
        VALIDATION_RESULT_DIR, "validation", signature, outputs=outputs
    )
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
        ("null calibration", ("fpr", "welch_moments"), _compute_null_calibration_tables),
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
            logger=LOGGER,
            label=f"Validation {label}",
        ):
            continue
        tables = compute()
        if set(tables) != set(keys):
            raise RuntimeError(f"Validation stage returned {sorted(tables)}, expected {keys}")
        for key, table in tables.items():
            table.to_csv(VALIDATION_RESULT_DIR / VALIDATION_CACHE_FILES[key], index=False)
        LOGGER.info("Validation %s: cached %d tables", label, len(tables))
    if force:
        update_cache_entry(
            VALIDATION_RESULT_DIR,
            "validation",
            signature,
            outputs,
            signature_data=signature_data,
        )
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
