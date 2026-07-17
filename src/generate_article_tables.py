"""Regenerate all numerical results associated with the article.

Two separate protocols are used.

1. Cumulative multiprecision trajectories:
   - 1200 decimal digits;
   - tolerance 1e-500;
   - three repetitions with alternating exponential/retraction order;
   - median cumulative time at each iteration and the MAD of total time.

   The residuals and times for each method belong to the same protocol.
   Therefore, if one method terminates earlier (for example, the retraction
   on S^3), no artificial rows are added after convergence.

2. Moderate-precision timing control:
   - 50 decimal digits;
   - tolerance 1e-14;
   - two warm-ups and eleven repetitions with alternating order;
   - median and MAD of total time.

The script writes results to the multiprecision, moderate_precision, and
diagnostics directories under the directory specified by --output-root.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import runpy
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mpmath as mp


BASE = Path(__file__).resolve().parent
FULL_DPS = 1200
FULL_TOL = mp.mpf("1e-500")
FULL_REPEATS = 3
FULL_WARMUP_DPS = 100
FULL_WARMUP_TOL = mp.mpf("1e-20")
MODERATE_DPS = 50
MODERATE_TOL = mp.mpf("1e-14")
MODERATE_WARMUPS = 2
MODERATE_REPEATS = 11
ORDER_THRESHOLD = mp.mpf("1e-300")


@dataclass
class MethodAdapter:
    name: str
    initial_state: Callable[[], Any]
    residual: Callable[[Any], mp.mpf]
    direction: Callable[[Any], tuple[Any, ...]]
    update: Callable[[Any, tuple[Any, ...], mp.mpf], Any]
    max_iter: int
    diagnostics: Callable[[], dict[str, Any]] | None = None


@dataclass
class Experiment:
    slug: str
    title: str
    exp_method: MethodAdapter
    ret_method: MethodAdapter
    special_order_exp: str | None = None
    special_order_ret: str | None = None


def load_module(filename: str) -> dict[str, Any]:
    return runpy.run_path(str(BASE / filename), run_name=f"module_{filename}")


def run_once(
    method: MethodAdapter,
    tolerance: mp.mpf,
    *,
    timed: bool,
) -> tuple[list[mp.mpf], list[float], list[dict[str, Any]]]:
    state = method.initial_state()
    residuals: list[mp.mpf] = []
    elapsed: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    start = time.perf_counter() if timed else 0.0

    for _k in range(method.max_iter + 1):
        residual = mp.mpf(method.residual(state))
        residuals.append(residual)
        elapsed.append(time.perf_counter() - start if timed else float("nan"))
        if residual <= tolerance:
            return residuals, elapsed, diagnostics

        step_data = method.direction(state)
        state = method.update(state, step_data, residual)
        diagnostic = dict(method.diagnostics() if method.diagnostics else {})
        diagnostic["residual_before_step"] = residual
        diagnostics.append(diagnostic)

    raise RuntimeError(f"{method.name}: did not reach the tolerance {tolerance}.")


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def estimate_order(residuals: list[mp.mpf]) -> mp.mpf | None:
    estimates: list[mp.mpf] = []
    for k in range(1, len(residuals) - 1):
        rm1, r0, rp1 = residuals[k - 1], residuals[k], residuals[k + 1]
        if min(rm1, r0, rp1) <= 0 or r0 <= ORDER_THRESHOLD:
            continue
        denominator = mp.log(r0 / rm1)
        if denominator == 0:
            continue
        rho = mp.log(rp1 / r0) / denominator
        if mp.isfinite(rho) and mp.mpf("0.5") < rho < mp.mpf("5"):
            estimates.append(rho)
    return estimates[-1] if estimates else None


def format_order(value: mp.mpf | None) -> str:
    return mp.nstr(value, 10) if value is not None else "not estimable"


def _validate_residuals(reference: list[mp.mpf], candidate: list[mp.mpf], name: str) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(f"{name}: different iteration counts across repetitions.")
    threshold = mp.power(10, -min(500, max(30, mp.mp.dps // 2)))
    for k, (a, b) in enumerate(zip(reference, candidate)):
        if abs(a - b) > threshold * max(mp.mpf(1), abs(a), abs(b)):
            raise RuntimeError(f"{name}: nonreproducible trajectory at k={k}.")


def full_benchmark_pair(
    exp_method: MethodAdapter,
    ret_method: MethodAdapter,
) -> tuple[
    list[mp.mpf], list[mp.mpf], list[float], list[float], float, float,
    list[dict[str, Any]],
]:
    # Short warm-up to load code paths and special functions.
    mp.mp.dps = FULL_WARMUP_DPS
    for method in (exp_method, ret_method):
        run_once(method, FULL_WARMUP_TOL, timed=True)

    mp.mp.dps = FULL_DPS
    runs: dict[str, list[tuple[list[mp.mpf], list[float], list[dict[str, Any]]]]] = {
        "exp": [], "ret": []
    }
    methods = {"exp": exp_method, "ret": ret_method}

    for repetition in range(FULL_REPEATS):
        keys = ("exp", "ret") if repetition % 2 == 0 else ("ret", "exp")
        for key in keys:
            print(
                f"    multiprecision repetition {repetition + 1}/{FULL_REPEATS}: "
                f"{methods[key].name}",
                flush=True,
            )
            runs[key].append(run_once(methods[key], FULL_TOL, timed=True))

    output: dict[str, tuple[list[mp.mpf], list[float], float]] = {}
    first_diagnostics: list[dict[str, Any]] = runs["exp"][0][2]
    for key in ("exp", "ret"):
        reference = runs[key][0][0]
        for residuals, _times, _diagnostics in runs[key][1:]:
            _validate_residuals(reference, residuals, methods[key].name)
        cumulative = [
            statistics.median(run[1][k] for run in runs[key])
            for k in range(len(reference))
        ]
        final_times = [run[1][-1] for run in runs[key]]
        output[key] = (
            reference,
            cumulative,
            median_absolute_deviation(final_times),
        )

    return (
        output["exp"][0], output["ret"][0],
        output["exp"][1], output["ret"][1],
        output["exp"][2], output["ret"][2],
        first_diagnostics,
    )


def moderate_benchmark_pair(
    exp_method: MethodAdapter,
    ret_method: MethodAdapter,
) -> dict[str, float | int]:
    mp.mp.dps = MODERATE_DPS
    methods = {"exp": exp_method, "ret": ret_method}

    for warmup in range(MODERATE_WARMUPS):
        keys = ("exp", "ret") if warmup % 2 == 0 else ("ret", "exp")
        for key in keys:
            run_once(methods[key], MODERATE_TOL, timed=True)

    totals: dict[str, list[float]] = {"exp": [], "ret": []}
    iterations: dict[str, int] = {}
    for repetition in range(MODERATE_REPEATS):
        keys = ("exp", "ret") if repetition % 2 == 0 else ("ret", "exp")
        for key in keys:
            residuals, elapsed, _ = run_once(
                methods[key], MODERATE_TOL, timed=True
            )
            totals[key].append(elapsed[-1])
            iterations[key] = len(residuals) - 1

    med_exp = statistics.median(totals["exp"])
    med_ret = statistics.median(totals["ret"])
    return {
        "exp_iterations": iterations["exp"],
        "ret_iterations": iterations["ret"],
        "exp_median_s": med_exp,
        "exp_mad_s": median_absolute_deviation(totals["exp"]),
        "ret_median_s": med_ret,
        "ret_mad_s": median_absolute_deviation(totals["ret"]),
        "reduction_percent": 100.0 * (1.0 - med_ret / med_exp),
    }


def write_sl2_diagnostics(
    diagnostics: list[dict[str, Any]],
    path: Path,
) -> None:
    fields = [
        "step",
        "residual_before_step",
        "requested_tolerance",
        "series_order",
        "a_posteriori_tail_estimate",
        "recent_max_ratio",
        "raw_determinant_defect",
        "stabilization_norm",
        "final_determinant_defect",
        "raw_tangency_defect",
        "raw_geodesic_residual",
        "raw_energy_defect",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for k, d in enumerate(diagnostics):
            writer.writerow({
                "step": k,
                "residual_before_step": mp.nstr(d.get("residual_before_step", ""), 30),
                "requested_tolerance": mp.nstr(d.get("requested_tolerance", ""), 30),
                "series_order": d.get("order", ""),
                "a_posteriori_tail_estimate": mp.nstr(d.get("tail_bound", ""), 30),
                "recent_max_ratio": mp.nstr(d.get("ratio_bound", ""), 30),
                "raw_determinant_defect": mp.nstr(d.get("raw_determinant_defect", ""), 30),
                "stabilization_norm": mp.nstr(d.get("stabilization_norm", ""), 30),
                "final_determinant_defect": mp.nstr(d.get("determinant_defect", ""), 30),
                "raw_tangency_defect": mp.nstr(d.get("tangency_defect_raw", ""), 30),
                "raw_geodesic_residual": mp.nstr(d.get("geodesic_residual_raw", ""), 30),
                "raw_energy_defect": mp.nstr(d.get("energy_defect_raw", ""), 30),
            })


def write_experiment(
    experiment: Experiment,
    multiprecision_dir: Path,
    moderate_dir: Path,
    diagnostics_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    print(f"Running {experiment.title}...", flush=True)
    (
        residuals_exp, residuals_ret,
        times_exp, times_ret,
        mad_exp, mad_ret,
        exp_diagnostics,
    ) = full_benchmark_pair(experiment.exp_method, experiment.ret_method)

    order_exp = experiment.special_order_exp or format_order(
        estimate_order(residuals_exp)
    )
    order_ret = experiment.special_order_ret or format_order(
        estimate_order(residuals_ret)
    )

    table_path = multiprecision_dir / f"table_{experiment.slug}.csv"
    nrows = max(len(residuals_exp), len(residuals_ret))
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "k", "exp_residual", "ret_residual",
                "exp_cumulative_time_s", "ret_cumulative_time_s",
            ],
        )
        writer.writeheader()
        for k in range(nrows):
            writer.writerow({
                "k": k,
                "exp_residual": mp.nstr(residuals_exp[k], 50) if k < len(residuals_exp) else "",
                "ret_residual": mp.nstr(residuals_ret[k], 50) if k < len(residuals_ret) else "",
                "exp_cumulative_time_s": f"{times_exp[k]:.12f}" if k < len(times_exp) else "",
                "ret_cumulative_time_s": f"{times_ret[k]:.12f}" if k < len(times_ret) else "",
            })

    final_exp = times_exp[-1]
    final_ret = times_ret[-1]
    full_summary = {
        "experiment": experiment.title,
        "file": table_path.name,
        "exp_iterations": str(len(residuals_exp) - 1),
        "ret_iterations": str(len(residuals_ret) - 1),
        "exp_order": order_exp,
        "ret_order": order_ret,
        "exp_final_time_s": f"{final_exp:.12f}",
        "exp_final_mad_s": f"{mad_exp:.12f}",
        "ret_final_time_s": f"{final_ret:.12f}",
        "ret_final_mad_s": f"{mad_ret:.12f}",
        "reduction_percent": f"{100.0 * (1.0 - final_ret / final_exp):.6f}",
    }

    print(f"    moderate-precision timings: {experiment.title}", flush=True)
    moderate = moderate_benchmark_pair(experiment.exp_method, experiment.ret_method)
    moderate_summary = {
        "experiment": experiment.title,
        **{key: str(value) for key, value in moderate.items()},
    }

    if experiment.slug == "sl2_nonlinear":
        write_sl2_diagnostics(
            exp_diagnostics,
            diagnostics_dir / "sl2_geodesic_diagnostics.csv",
        )

    return full_summary, moderate_summary


def build_experiments() -> list[Experiment]:
    m = load_module("experiment_leslie_regular_cylinder.py")
    p0_cyl = lambda: mp.matrix([mp.mpf("4") / 5, mp.mpf("1") / 2, mp.mpf("3") / 5])
    dir_cyl = lambda p: m["leslie_newton_direction"](p)
    exp_cyl = MethodAdapter(
        "Cylinder + exponential", p0_cyl,
        lambda p: m["norm"](m["field"](p)), dir_cyl,
        lambda p, data, residual: m["exponential_update"](p, data[0]), m["MAX_ITER"],
    )
    ret_cyl = MethodAdapter(
        "Cylinder + retraction", p0_cyl,
        lambda p: m["norm"](m["field"](p)), dir_cyl,
        lambda p, data, residual: m["retraction_update"](p, data[0]), m["MAX_ITER"],
    )

    m2 = load_module("experiment_sphere_s3_exp_retr.py")
    def p0_s3() -> mp.matrix:
        s3 = mp.sqrt(3)
        return mp.matrix([s3 / 2, 1 / (2 * s3), 1 / (2 * s3), 1 / (2 * s3)])
    dir_s3 = lambda p: m2["newton_direction"](p)
    exp_s3 = MethodAdapter(
        "S3 + exponential", p0_s3,
        lambda p: m2["norm"](m2["field"](p)), dir_s3,
        lambda p, data, residual: m2["exponential_update"](p, data[0]), m2["MAX_ITER"],
    )
    ret_s3 = MethodAdapter(
        "S3 + retraction", p0_s3,
        lambda p: m2["norm"](m2["field"](p)), dir_s3,
        lambda p, data, residual: m2["retraction_update"](p, data[0]), m2["MAX_ITER"],
    )

    m3 = load_module("experiment_product_torus_exp_retr.py")
    p0_torus = lambda: m3["point_from_angles"](m3["THETA0"], m3["PHI0"])
    dir_torus = lambda p: m3["newton_direction"](p)
    exp_torus = MethodAdapter(
        "Product torus + exponential", p0_torus,
        lambda p: m3["norm"](m3["field"](p)), dir_torus,
        lambda p, data, residual: m3["exponential_update"](p, data[0], data[2]), m3["MAX_ITER"],
    )
    ret_torus = MethodAdapter(
        "Product torus + retraction", p0_torus,
        lambda p: m3["norm"](m3["field"](p)), dir_torus,
        lambda p, data, residual: m3["retraction_update"](p, data[0], data[2]), m3["MAX_ITER"],
    )

    m4 = load_module("experiment_sl2_riemannian_exponential_homogeneous_retraction.py")
    def a0_sl2() -> mp.matrix:
        return mp.matrix([[mp.mpf("1.2"), mp.mpf("0.4")],
                          [mp.mpf("0.2"), mp.mpf("0.9")]])
    dir_sl2 = lambda A: m4["newton_direction"](A)
    exp_sl2 = MethodAdapter(
        "SL(2,R) + Riemannian exponential", a0_sl2,
        lambda A: m4["frobenius_norm"](m4["field"](A)), dir_sl2,
        lambda A, data, residual: m4["riemannian_exponential"](A, data[0], residual),
        m4["MAX_ITER"],
        diagnostics=lambda: m4["LAST_GEODESIC_DIAGNOSTICS"],
    )
    det_sl2 = MethodAdapter(
        "SL(2,R) + homogeneous retraction", a0_sl2,
        lambda A: m4["frobenius_norm"](m4["field"](A)), dir_sl2,
        lambda A, data, residual: m4["determinant_retraction"](A, data[0]), m4["MAX_ITER"],
    )

    m5 = load_module("experiment_st32_exponential_polar.py")
    p0_st = lambda: m5["initial_frame"]()
    dir_st = lambda Y: m5["newton_direction"](Y)
    exp_st = MethodAdapter(
        "Nonlinear Stiefel + exponential", p0_st,
        lambda Y: m5["frobenius_norm"](m5["field"](Y)), dir_st,
        lambda Y, data, residual: m5["euclidean_exponential"](Y, data[0], residual), m5["MAX_ITER"],
    )
    ret_st = MethodAdapter(
        "Nonlinear Stiefel + polar retraction", p0_st,
        lambda Y: m5["frobenius_norm"](m5["field"](Y)), dir_st,
        lambda Y, data, residual: m5["polar_retraction"](Y, data[0]), m5["MAX_ITER"],
    )

    m6 = load_module("experiment_wahba.py")
    p0_wahba = lambda: m6["initial_quaternion"]()
    dir_wahba = lambda q: m6["newton_direction"](q)
    exp_wahba = MethodAdapter(
        "Wahba + exponential", p0_wahba,
        lambda q: m6["norm"](m6["field"](q)), dir_wahba,
        lambda q, data, residual: m6["exponential_update"](q, data[0]), m6["MAX_ITER"],
    )
    ret_wahba = MethodAdapter(
        "Wahba + normalization retraction", p0_wahba,
        lambda q: m6["norm"](m6["field"](q)), dir_wahba,
        lambda q, data, residual: m6["normalization_retraction"](q, data[0]), m6["MAX_ITER"],
    )

    m7 = load_module("experiment_toroidal_von_mises.py")
    p0_vm = lambda: m7["point_from_angles"](m7["THETA0"], m7["PHI0"])
    dir_vm = lambda p: m7["newton_direction"](p)
    exp_vm = MethodAdapter(
        "Von Mises + exponential", p0_vm,
        lambda p: m7["norm"](m7["field"](p)), dir_vm,
        lambda p, data, residual: m7["exponential_update"](p, data[0], data[2]), m7["MAX_ITER"],
    )
    ret_vm = MethodAdapter(
        "Von Mises + normalization retraction", p0_vm,
        lambda p: m7["norm"](m7["field"](p)), dir_vm,
        lambda p, data, residual: m7["normalization_retraction"](p, data[0], data[2]), m7["MAX_ITER"],
    )

    m8 = load_module("experiment_brockett_stiefel.py")
    p0_brockett = lambda: mp.matrix(m8["Y0_FRAME"])
    dir_brockett = lambda Y: m8["newton_direction"](Y)
    exp_brockett = MethodAdapter(
        "Brockett + exponential", p0_brockett,
        lambda Y: m8["frobenius_norm"](m8["field"](Y)), dir_brockett,
        lambda Y, data, residual: m8["euclidean_exponential"](Y, data[0], residual), m8["MAX_ITER"],
    )
    ret_brockett = MethodAdapter(
        "Brockett + polar retraction", p0_brockett,
        lambda Y: m8["frobenius_norm"](m8["field"](Y)), dir_brockett,
        lambda Y, data, residual: m8["polar_retraction"](Y, data[0]), m8["MAX_ITER"],
    )

    return [
        Experiment("cylinder", "Cylinder", exp_cyl, ret_cyl),
        Experiment("sphere_s3", "Sphere S3", exp_s3, ret_s3,
                   "approx. 3", "finite termination (1 step)"),
        Experiment("product_torus", "Product torus", exp_torus, ret_torus),
        Experiment("sl2_nonlinear", "Nonlinear SL(2,R)", exp_sl2, det_sl2),
        Experiment("stiefel_nonlinear", "Nonlinear Stiefel field", exp_st, ret_st),
        Experiment("wahba", "Wahba on S3", exp_wahba, ret_wahba),
        Experiment("von_mises", "Toroidal von Mises model", exp_vm, ret_vm),
        Experiment("brockett", "Brockett on St(3,2)", exp_brockett, ret_brockett),
    ]


def cpu_description() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "not reported by the system"


def write_environment(root: Path) -> None:
    data = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.replace("\n", " "),
        "mpmath": mp.__version__,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": cpu_description(),
        "multiprecision": {
            "dps": FULL_DPS,
            "tolerance": str(FULL_TOL),
            "repetitions": FULL_REPEATS,
            "method_order": "alternating",
            "statistic": "median cumulative time per iteration and final MAD",
        },
        "moderate_precision_timings": {
            "dps": MODERATE_DPS,
            "tolerance": str(MODERATE_TOL),
            "warmups": MODERATE_WARMUPS,
            "repetitions": MODERATE_REPEATS,
            "method_order": "alternating",
            "statistic": "median and MAD of total time",
        },
        "clock": "time.perf_counter",
    }
    (root / "execution_environment.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE.parent / "results_local",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    multiprecision_dir = root / "multiprecision"
    moderate_dir = root / "moderate_precision"
    diagnostics_dir = root / "diagnostics"
    for directory in (root, multiprecision_dir, moderate_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_environment(root)

    full_rows: list[dict[str, str]] = []
    moderate_rows: list[dict[str, str]] = []
    for experiment in build_experiments():
        full, moderate = write_experiment(
            experiment, multiprecision_dir, moderate_dir, diagnostics_dir
        )
        full_rows.append(full)
        moderate_rows.append(moderate)

    with (multiprecision_dir / "multiprecision_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(full_rows[0].keys()))
        writer.writeheader()
        writer.writerows(full_rows)

    with (moderate_dir / "moderate_precision_timing_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(moderate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(moderate_rows)

    print("\nMULTIPRECISION SUMMARY", flush=True)
    for row in full_rows:
        print(
            f"{row['experiment']}: "
            f"Exp={row['exp_final_time_s']} s, "
            f"R={row['ret_final_time_s']} s, "
            f"reduction={row['reduction_percent']}%",
            flush=True,
        )
    print(f"\nResults written to {root}", flush=True)


if __name__ == "__main__":
    main()
