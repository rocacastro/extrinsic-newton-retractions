"""
Extrinsic Newton method on SL(2,R) with the induced Frobenius metric.

The following updates are compared:
    1. the Riemannian exponential Exp_A^{SL}(eta), obtained by integrating
       the extrinsic geodesic equation;
    2. the homogeneous retraction
           R_A^{det}(eta)=det(A+eta)^(-1/2)(A+eta).

For gamma(t) in SL(2,R), V=dot(gamma), and B=gamma^{-1}V, a geodesic of the
induced metric satisfies

    ddot(gamma)
      = tr(B^2)/||gamma^{-1}||_F^2 gamma^{-T}.

The implementation uses a Taylor series computed recursively from the
equivalent form

    ddot(gamma) = lambda cof(gamma),
    lambda = -<cof(V),V>_F/||gamma||_F^2,

which is valid for 2x2 matrices of determinant one. Truncation is controlled
by an a posteriori criterion based on coefficient ratios. The geodesic
tolerance adapts to the Newton residual: early steps are not computed at
unnecessarily extreme accuracy, whereas later steps receive the precision
required to reach 1e-500. After each integration, determinant stabilization
of the same scale as the numerical integration error is applied. The group
exponential is not used.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import mpmath as mp


mp.mp.dps = 1200
TOL = mp.mpf("1e-500")
MAX_ITER = 40
OUTPUT_CSV = Path(
    "sl2_riemannian_exponential_homogeneous_retraction_detailed_table.csv"
)

E1 = mp.matrix([[1, 0], [0, -1]])
E2 = mp.matrix([[0, 1], [0, 0]])
E3 = mp.matrix([[0, 0], [1, 0]])
SL2_BASIS = (E1, E2, E3)


def trace(A: mp.matrix) -> mp.mpf:
    return A[0, 0] + A[1, 1]


def frobenius_inner(A: mp.matrix, B: mp.matrix) -> mp.mpf:
    return mp.fsum(
        A[i, j] * B[i, j]
        for i in range(2)
        for j in range(2)
    )


def frobenius_norm(A: mp.matrix) -> mp.mpf:
    return mp.sqrt(frobenius_inner(A, A))


def det2(A: mp.matrix) -> mp.mpf:
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def cofactor(A: mp.matrix) -> mp.matrix:
    return mp.matrix([
        [A[1, 1], -A[1, 0]],
        [-A[0, 1], A[0, 0]],
    ])


def sci(value: mp.mpf, digits: int = 18) -> str:
    if value == 0:
        return "0"
    return mp.nstr(value, digits)


def verification_tolerance() -> mp.mpf:
    exponent = min(900, max(20, mp.mp.dps - 20))
    return mp.power(10, -exponent)


def adaptive_geodesic_tolerance(residual: mp.mpf | None = None) -> mp.mpf:
    """Absolute tolerance for geodesic integration.

    In the multiprecision protocol, at most 1e-80 is used during early steps,
    and the tolerance is reduced proportionally to residual**3 down to
    1e-560. In the 50-digit timing protocol, an analogous range between
    1e-20 and 1e-40 is used. This avoids requiring 500 correct digits for a
    geodesic while the iterate is still far from the zero.
    """
    if mp.mp.dps >= 600:
        floor = mp.mpf("1e-560")
        ceiling = mp.mpf("1e-80")
    else:
        floor = mp.power(10, -(max(20, mp.mp.dps - 10)))
        ceiling = mp.power(10, -(max(12, min(20, mp.mp.dps // 2))))

    if residual is None or residual <= 0:
        return floor
    return max(floor, min(ceiling, mp.mpf(residual) ** 3))


def tangent_project(A: mp.matrix, Z: mp.matrix) -> mp.matrix:
    Ainv = mp.inverse(A)
    normal = Ainv.T
    numerator = trace(Ainv * Z)
    denominator = frobenius_inner(normal, normal)
    return Z - (numerator / denominator) * normal


def tangent_basis(A: mp.matrix) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    return tuple(A * E for E in SL2_BASIS)


def scalar_coefficients(A: mp.matrix) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    g1 = a * a + b * c + b * b - 2
    g2 = b * b + c * d + a * c - 1
    g3 = c * c + a * b + d * d - 1
    return g1, g2, g3


def S_matrix(A: mp.matrix) -> mp.matrix:
    g1, g2, g3 = scalar_coefficients(A)
    return mp.matrix([[g1, g2], [g3, -g1]])


def field(A: mp.matrix) -> mp.matrix:
    return A * S_matrix(A)


def derivative_S(A: mp.matrix, H: mp.matrix) -> mp.matrix:
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    ha, hb = H[0, 0], H[0, 1]
    hc, hd = H[1, 0], H[1, 1]

    dg1 = 2 * a * ha + (c + 2 * b) * hb + b * hc
    dg2 = c * ha + 2 * b * hb + (d + a) * hc + c * hd
    dg3 = b * ha + a * hb + 2 * c * hc + 2 * d * hd
    return mp.matrix([[dg1, dg2], [dg3, -dg1]])


def ambient_derivative_field(A: mp.matrix, H: mp.matrix) -> mp.matrix:
    return H * S_matrix(A) + A * derivative_S(A, H)


def covariant_derivative_field(A: mp.matrix, H: mp.matrix) -> mp.matrix:
    return tangent_project(A, ambient_derivative_field(A, H))


def newton_direction(
    A: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    basis = tangent_basis(A)
    X = field(A)
    M = mp.matrix(3, 3)
    rhs = mp.matrix(3, 1)
    derivatives = [covariant_derivative_field(A, Bj) for Bj in basis]

    for i in range(3):
        rhs[i] = -frobenius_inner(basis[i], X)
        for j in range(3):
            M[i, j] = frobenius_inner(basis[i], derivatives[j])

    xi = mp.lu_solve(M, rhs)
    eta = mp.zeros(2, 2)
    for j in range(3):
        eta += xi[j] * basis[j]

    check = verification_tolerance()
    tangent_defect = abs(trace(mp.inverse(A) * eta))
    newton_defect = frobenius_norm(
        covariant_derivative_field(A, eta) + X
    )
    if tangent_defect > check:
        raise ArithmeticError("The Newton direction is not tangent.")
    if newton_defect > check:
        raise ArithmeticError("The Newton equation was not solved.")
    return eta, M, xi


# Series are stored as tuples (a11,a12,a21,a22) to avoid the cost
# instead of creating matrices in every convolution.
SeriesVector = tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]


def _to_vector(A: mp.matrix) -> SeriesVector:
    return (A[0, 0], A[0, 1], A[1, 0], A[1, 1])


def _to_matrix(x: SeriesVector) -> mp.matrix:
    return mp.matrix([[x[0], x[1]], [x[2], x[3]]])


def _add(x: SeriesVector, y: SeriesVector) -> SeriesVector:
    return tuple(x[i] + y[i] for i in range(4))  # type: ignore[return-value]


def _scale(c: mp.mpf, x: SeriesVector) -> SeriesVector:
    return tuple(c * x[i] for i in range(4))  # type: ignore[return-value]


def _cofactor_vector(x: SeriesVector) -> SeriesVector:
    return (x[3], -x[2], -x[1], x[0])


def _inner_vector(x: SeriesVector, y: SeriesVector) -> mp.mpf:
    return mp.fsum(x[i] * y[i] for i in range(4))


def _norm_vector(x: SeriesVector) -> mp.mpf:
    return mp.sqrt(_inner_vector(x, x))


def geodesic_rhs(A: mp.matrix, V: mp.matrix) -> mp.matrix:
    Ainv = mp.inverse(A)
    B = Ainv * V
    coefficient = trace(B * B) / frobenius_inner(Ainv, Ainv)
    return coefficient * Ainv.T


def _taylor_geodesic(
    A: mp.matrix,
    eta: mp.matrix,
    tolerance: mp.mpf,
    max_order: int = 3000,
) -> tuple[mp.matrix, mp.matrix, mp.matrix, dict[str, mp.mpf | int]]:
    a0 = _to_vector(A)
    a1 = _to_vector(eta)
    coefficients: list[SeriesVector] = [a0, a1]
    denominator_coefficients: list[mp.mpf] = []
    lambda_coefficients: list[mp.mpf] = []
    recent_norms: list[mp.mpf] = []
    zero: SeriesVector = (mp.mpf("0"),) * 4
    position_sum = _add(a0, a1)
    denominator_zero = _inner_vector(a0, a0)

    tail_bound = mp.inf
    ratio_bound = mp.inf

    for n in range(max_order - 1):
        denominator_n = mp.fsum(
            _inner_vector(coefficients[i], coefficients[n - i])
            for i in range(n + 1)
        )
        denominator_coefficients.append(denominator_n)

        numerator_n = -mp.fsum(
            _inner_vector(
                _cofactor_vector(_scale(i + 1, coefficients[i + 1])),
                _scale(n - i + 1, coefficients[n - i + 1]),
            )
            for i in range(n + 1)
        )

        convolution = mp.fsum(
            denominator_coefficients[i] * lambda_coefficients[n - i]
            for i in range(1, n + 1)
        )
        lambda_n = (numerator_n - convolution) / denominator_zero
        lambda_coefficients.append(lambda_n)

        acceleration_n = zero
        for i in range(n + 1):
            acceleration_n = _add(
                acceleration_n,
                _scale(
                    lambda_coefficients[i],
                    _cofactor_vector(coefficients[n - i]),
                ),
            )

        next_coefficient = _scale(
            mp.mpf(1) / ((n + 2) * (n + 1)),
            acceleration_n,
        )
        coefficients.append(next_coefficient)
        position_sum = _add(position_sum, next_coefficient)

        coefficient_norm = _norm_vector(next_coefficient)
        recent_norms.append(coefficient_norm)

        if n > 30 and len(recent_norms) >= 10:
            if coefficient_norm == 0:
                tail_bound = mp.mpf("0")
                ratio_bound = mp.mpf("0")
            else:
                ratios = [
                    recent_norms[-j] / recent_norms[-j - 1]
                    for j in range(1, 9)
                    if recent_norms[-j - 1] != 0
                ]
                ratio_bound = max(ratios) if ratios else mp.inf
                if ratio_bound < 1:
                    tail_bound = coefficient_norm / (1 - ratio_bound)

            if ratio_bound < mp.mpf("0.9") and tail_bound < tolerance:
                velocity_sum = zero
                acceleration_sum = zero
                for k in range(1, len(coefficients)):
                    velocity_sum = _add(
                        velocity_sum,
                        _scale(k, coefficients[k]),
                    )
                    if k >= 2:
                        acceleration_sum = _add(
                            acceleration_sum,
                            _scale(k * (k - 1), coefficients[k]),
                        )

                endpoint = _to_matrix(position_sum)
                endpoint_velocity = _to_matrix(velocity_sum)
                endpoint_acceleration = _to_matrix(acceleration_sum)
                diagnostics: dict[str, mp.mpf | int] = {
                    "order": len(coefficients) - 1,
                    "tail_bound": tail_bound,
                    "ratio_bound": ratio_bound,
                }
                return (
                    endpoint,
                    endpoint_velocity,
                    endpoint_acceleration,
                    diagnostics,
                )

    raise ArithmeticError(
        "The geodesic series did not reach the requested tolerance."
    )


LAST_GEODESIC_DIAGNOSTICS: dict[str, mp.mpf | int] = {}


def riemannian_exponential(
    A: mp.matrix,
    eta: mp.matrix,
    residual: mp.mpf | None = None,
) -> mp.matrix:
    """Evaluate the Riemannian exponential on SL(2,R) numerically.

    The geodesic equation is integrated using a recursive Taylor series. The
    tolerance adapts to the field residual. Final determinant normalization
    corrects only the numerical integration error; its magnitude is recorded
    in LAST_GEODESIC_DIAGNOSTICS.
    """
    global LAST_GEODESIC_DIAGNOSTICS

    tangent_initial = abs(trace(mp.inverse(A) * eta))
    if tangent_initial > verification_tolerance():
        raise ArithmeticError("The initial velocity is not tangent.")

    tolerance = adaptive_geodesic_tolerance(residual)
    endpoint_raw, velocity, acceleration, diagnostics = _taylor_geodesic(
        A,
        eta,
        tolerance,
    )

    determinant_raw = det2(endpoint_raw)
    if determinant_raw <= 0:
        raise ArithmeticError(
            "Geodesic integration produced a nonpositive determinant."
        )

    endpoint = endpoint_raw / mp.sqrt(determinant_raw)
    stabilization = frobenius_norm(endpoint - endpoint_raw)

    rhs_endpoint = geodesic_rhs(endpoint_raw, velocity)
    initial_energy = frobenius_inner(eta, eta)
    final_energy = frobenius_inner(velocity, velocity)
    diagnostics.update({
        "requested_tolerance": tolerance,
        "raw_determinant_defect": abs(determinant_raw - 1),
        "stabilization_norm": stabilization,
        "determinant_defect": abs(det2(endpoint) - 1),
        "tangency_defect_raw": abs(
            trace(mp.inverse(endpoint_raw) * velocity)
        ),
        "geodesic_residual_raw": frobenius_norm(
            acceleration - rhs_endpoint
        ),
        "energy_defect_raw": abs(final_energy - initial_energy),
        "initial_position_defect": mp.mpf("0"),
        "initial_velocity_defect": mp.mpf("0"),
    })
    LAST_GEODESIC_DIAGNOSTICS.clear()
    LAST_GEODESIC_DIAGNOSTICS.update(diagnostics)

    # Conservative check. It is not intended to certify the entire
    # series tail rigorously, but to detect a clearly failed integration.
    check = max(mp.sqrt(tolerance), mp.power(10, -max(10, mp.mp.dps // 4)))
    for key in (
        "raw_determinant_defect",
        "tangency_defect_raw",
        "geodesic_residual_raw",
        "energy_defect_raw",
    ):
        if mp.mpf(diagnostics[key]) > check:
            raise ArithmeticError(
                f"Geodesic diagnostic failed for {key}: {diagnostics[key]}"
            )
    return endpoint


def determinant_retraction(A: mp.matrix, eta: mp.matrix) -> mp.matrix:
    B = A + eta
    determinant = det2(B)
    if determinant <= 0:
        raise ArithmeticError("det(A+eta) is not positive.")
    return B / mp.sqrt(determinant)


def estimate_orders(raw: list[dict]) -> None:
    for j, row in enumerate(raw):
        row["order"] = mp.nan
        if 1 <= j < len(raw) - 1:
            em1 = raw[j - 1]["residual"]
            e0 = raw[j]["residual"]
            ep1 = raw[j + 1]["residual"]
            if em1 > 0 and e0 > 0 and ep1 > 0 and em1 != e0:
                row["order"] = mp.log(ep1 / e0) / mp.log(e0 / em1)


def run(
    update: Callable[[mp.matrix, mp.matrix], mp.matrix],
    method_name: str,
    tolerance: mp.mpf = TOL,
) -> list[dict[str, str]]:
    A = mp.matrix([
        [mp.mpf("1.2"), mp.mpf("0.4")],
        [mp.mpf("0.2"), mp.mpf("0.9")],
    ])
    raw: list[dict] = []

    for k in range(MAX_ITER + 1):
        X = field(A)
        residual = frobenius_norm(X)
        eta, M, _ = newton_direction(A)
        diagnostics = dict(LAST_GEODESIC_DIAGNOSTICS)
        raw.append({
            "method": method_name,
            "iteration": k,
            "A": mp.matrix(A),
            "residual": residual,
            "det_defect": abs(det2(A) - 1),
            "det_M": mp.det(M),
            "step_norm": frobenius_norm(eta),
            "geodesic": diagnostics,
        })
        if residual <= tolerance:
            break
        A = update(A, eta)
    else:
        raise RuntimeError(f"{method_name}: tolerance was not reached.")

    A_star = raw[-1]["A"]
    for row in raw:
        row["error"] = frobenius_norm(row["A"] - A_star)
    estimate_orders(raw)

    rows: list[dict[str, str]] = []
    for row in raw:
        Arow = row["A"]
        g1, g2, g3 = scalar_coefficients(Arow)
        rows.append({
            "method": row["method"],
            "iteration": str(row["iteration"]),
            "a11": sci(Arow[0, 0], 50),
            "a12": sci(Arow[0, 1], 50),
            "a21": sci(Arow[1, 0], 50),
            "a22": sci(Arow[1, 1], 50),
            "g1": sci(g1, 50),
            "g2": sci(g2, 50),
            "g3": sci(g3, 50),
            "residual": sci(row["residual"], 50),
            "frobenius_error": sci(row["error"], 50),
            "determinant_defect": sci(row["det_defect"], 30),
            "det_M": sci(row["det_M"], 50),
            "step_norm": sci(row["step_norm"], 50),
            "estimated_order": (
                "" if mp.isnan(row["order"]) else sci(row["order"], 20)
            ),
        })
    return rows


def main() -> None:
    rows_exp = run(riemannian_exponential, "Newton + Riemannian exponential")
    rows_ret = run(determinant_retraction, "Newton + homogeneous retraction")
    all_rows = rows_exp + rows_ret

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    last_exp = rows_exp[-1]
    last_ret = rows_ret[-1]
    root_difference = frobenius_norm(mp.matrix([
        [mp.mpf(last_exp["a11"]), mp.mpf(last_exp["a12"])],
        [mp.mpf(last_exp["a21"]), mp.mpf(last_exp["a22"])],
    ]) - mp.matrix([
        [mp.mpf(last_ret["a11"]), mp.mpf(last_ret["a12"])],
        [mp.mpf(last_ret["a21"]), mp.mpf(last_ret["a22"])],
    ]))

    if root_difference > mp.mpf("1e-40"):
        raise ArithmeticError("The two updates did not reach the same root.")
    if abs(mp.mpf(last_exp["det_M"])) < mp.mpf("1e-20"):
        raise ArithmeticError("The computed root is not regular.")

    for rows, label in ((rows_exp, "Exp"), (rows_ret, "R")):
        orders = [
            mp.mpf(row["estimated_order"])
            for row in rows
            if row["estimated_order"]
        ]
        if not orders or not (mp.mpf("1.8") < orders[-1] < mp.mpf("2.2")):
            raise ArithmeticError(f"{label}: quadratic convergence was not verified.")

    print("SL(2,R): Riemannian exponential vs. homogeneous retraction")
    print("Iterations:", last_exp["iteration"], "(Exp),", last_ret["iteration"], "(R)")
    print("Difference between roots:", sci(root_difference, 20))
    print("det M(A*) with Exp:", last_exp["det_M"])
    print("Diagnostics for the final geodesic:")
    for key, value in LAST_GEODESIC_DIAGNOSTICS.items():
        print(f"  {key}: {value}")
    print("CSV:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
