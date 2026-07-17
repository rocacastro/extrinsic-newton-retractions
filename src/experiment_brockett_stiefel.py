"""
Extrinsic Newton method for the Brockett eigenvector problem on St(3,2).

Application:
    Computation of an ordered orthonormal frame of eigenvectors of a
    symmetric matrix.

Ambient space:
    R^{3x2} ≅ R^6.

Manifold:
    St(3,2) = {Y in R^{3x2}: Y^T Y=I_2}.

Brockett function:
    Phi(Y)=1/2 tr(N Y^T A Y),
with symmetric A and N=diag(1,2).

Field:
    X(Y)=P_Y(A Y N)
        =A Y N-Y sym(Y^T A Y N).

The zeros are ordered eigenvector frames of A, with the local ordering fixed
by N.

The following updates are compared:
    1. The Riemannian exponential for the induced Euclidean metric.
    2. The polar retraction.

Tolerance:
    1e-500.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import mpmath as mp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

mp.mp.dps = 1200
TOL = mp.mpf("1e-500")
MAX_ITER = 35
OUTPUT_CSV = Path("brockett_stiefel_table.csv")

I2 = mp.eye(2)
N_WEIGHT = mp.matrix([[1, 0], [0, 2]])

K1 = mp.matrix([
    [0, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
])

K2 = mp.matrix([
    [0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0],
])

K3 = mp.matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 0],
])

SO3_BASIS = (K1, K2, K3)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def trace(A: mp.matrix) -> mp.mpf:
    return mp.fsum(A[i, i] for i in range(min(A.rows, A.cols)))


def frobenius_inner(A: mp.matrix, B: mp.matrix) -> mp.mpf:
    return mp.fsum(
        A[i, j] * B[i, j]
        for i in range(A.rows)
        for j in range(A.cols)
    )


def frobenius_norm(A: mp.matrix) -> mp.mpf:
    return mp.sqrt(frobenius_inner(A, A))


def sym(A: mp.matrix) -> mp.matrix:
    return mp.mpf("0.5") * (A + A.T)


def sci(value: mp.mpf, digits: int = 18) -> str:
    if value == 0:
        return "0"
    return mp.nstr(value, digits)


def verification_tolerance() -> mp.mpf:
    """Tolerance consistent with the active mpmath precision."""
    return mp.power(10, -min(900, max(20, mp.mp.dps - 20)))


def block_matrix_2x2(
    A11: mp.matrix,
    A12: mp.matrix,
    A21: mp.matrix,
    A22: mp.matrix,
) -> mp.matrix:
    result = mp.matrix(4, 4)

    for i in range(2):
        for j in range(2):
            result[i, j] = A11[i, j]
            result[i, j + 2] = A12[i, j]
            result[i + 2, j] = A21[i, j]
            result[i + 2, j + 2] = A22[i, j]

    return result


def horizontal_concatenate(A: mp.matrix, B: mp.matrix) -> mp.matrix:
    result = mp.matrix(A.rows, A.cols + B.cols)

    for i in range(A.rows):
        for j in range(A.cols):
            result[i, j] = A[i, j]

        for j in range(B.cols):
            result[i, A.cols + j] = B[i, j]

    return result


def vertical_concatenate(A: mp.matrix, B: mp.matrix) -> mp.matrix:
    result = mp.matrix(A.rows + B.rows, A.cols)

    for i in range(A.rows):
        for j in range(A.cols):
            result[i, j] = A[i, j]

    for i in range(B.rows):
        for j in range(B.cols):
            result[A.rows + i, j] = B[i, j]

    return result


# ---------------------------------------------------------------------------
# Eigenproblem data
# ---------------------------------------------------------------------------

def rotation_x(angle: mp.mpf) -> mp.matrix:
    c = mp.cos(angle)
    s = mp.sin(angle)

    return mp.matrix([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def rotation_y(angle: mp.mpf) -> mp.matrix:
    c = mp.cos(angle)
    s = mp.sin(angle)

    return mp.matrix([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def rotation_z(angle: mp.mpf) -> mp.matrix:
    c = mp.cos(angle)
    s = mp.sin(angle)

    return mp.matrix([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def build_problem() -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    """
    A=Q diag(1,2,4) Q^T with nontrivial Q.
    The target root is Y_*=Q[:,0:2].
    """
    Q = (
        rotation_z(mp.mpf("0.50"))
        *
        rotation_y(-mp.mpf("0.40"))
        *
        rotation_x(mp.mpf("0.30"))
    )

    eigenvalues = mp.matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 4],
    ])

    A = Q * eigenvalues * Q.T
    Y_star = Q[:, 0:2]

    perturbation = (
        rotation_z(mp.mpf("0.15"))
        *
        rotation_y(-mp.mpf("0.12"))
        *
        rotation_x(mp.mpf("0.10"))
    )

    Y0 = perturbation * Y_star

    return A, Y_star, Y0


A_MATRIX, Y_STAR, Y0_FRAME = build_problem()


# ---------------------------------------------------------------------------
# Geometry of St(3,2)
# ---------------------------------------------------------------------------

def constraint_defect(Y: mp.matrix) -> mp.mpf:
    return frobenius_norm(Y.T * Y - I2)


def tangent_project(Y: mp.matrix, Z: mp.matrix) -> mp.matrix:
    return Z - Y * sym(Y.T * Z)


def tangent_basis(Y: mp.matrix) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    return tuple(K * Y for K in SO3_BASIS)


# ---------------------------------------------------------------------------
# Brockett field
# ---------------------------------------------------------------------------

def field(Y: mp.matrix) -> mp.matrix:
    G = A_MATRIX * Y * N_WEIGHT
    return tangent_project(Y, G)


def ambient_derivative_field(Y: mp.matrix, H: mp.matrix) -> mp.matrix:
    G = A_MATRIX * Y * N_WEIGHT
    S = sym(Y.T * G)

    dG = A_MATRIX * H * N_WEIGHT
    dS = sym(H.T * G + Y.T * dG)

    return dG - H * S - Y * dS


def covariant_derivative_field(Y: mp.matrix, H: mp.matrix) -> mp.matrix:
    return tangent_project(Y, ambient_derivative_field(Y, H))


# ---------------------------------------------------------------------------
# Reduced 3x3 system
# ---------------------------------------------------------------------------

def newton_direction(
    Y: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    basis = tangent_basis(Y)
    X = field(Y)

    M = mp.matrix(3, 3)
    rhs = mp.matrix(3, 1)

    derivatives = [
        covariant_derivative_field(Y, Bj)
        for Bj in basis
    ]

    for i in range(3):
        rhs[i] = -frobenius_inner(basis[i], X)

        for j in range(3):
            M[i, j] = frobenius_inner(
                basis[i],
                derivatives[j],
            )

    xi = mp.lu_solve(M, rhs)

    eta = mp.zeros(3, 2)

    for j in range(3):
        eta += xi[j] * basis[j]

    tangency_defect = frobenius_norm(
        Y.T * eta + eta.T * Y
    )

    newton_defect = frobenius_norm(
        covariant_derivative_field(Y, eta) + X
    )

    if tangency_defect > verification_tolerance():
        raise ArithmeticError("The direction is not tangent.")

    if newton_defect > verification_tolerance():
        raise ArithmeticError("The Newton equation was not solved.")

    return eta, M, xi


# ---------------------------------------------------------------------------
# Induced-metric exponential and polar retraction
# ---------------------------------------------------------------------------

def adaptive_exponential_tolerance(residual: mp.mpf | None = None) -> mp.mpf:
    """Absolute tolerance for block matrix exponentials."""
    if mp.mp.dps >= 600:
        floor = mp.mpf("1e-560")
        ceiling = mp.mpf("1e-80")
    else:
        floor = mp.power(10, -max(20, mp.mp.dps - 10))
        ceiling = mp.power(10, -max(12, min(20, mp.mp.dps // 2)))
    if residual is None or residual <= 0:
        return floor
    return max(floor, min(ceiling, mp.mpf(residual) ** 3))


def matrix_inf_norm(M: mp.matrix) -> mp.mpf:
    return max(
        mp.fsum(abs(M[i, j]) for j in range(M.cols))
        for i in range(M.rows)
    )


def matrix_exponential_taylor(M: mp.matrix, tolerance: mp.mpf) -> mp.matrix:
    """Multiprecision matrix exponential by scaling, Taylor expansion, and squaring."""
    n = M.rows
    if M.cols != n:
        raise ValueError("The exponential requires a square matrix.")
    norm_M = matrix_inf_norm(M)
    if norm_M == 0:
        return mp.eye(n)
    scaling = max(0, int(mp.ceil(mp.log(norm_M / mp.mpf("0.25"), 2))))
    B = M / mp.power(2, scaling)
    result = mp.eye(n)
    term = mp.eye(n)
    local_tolerance = tolerance / max(1, mp.power(2, scaling + 2))
    for k in range(1, 5000):
        term = (term * B) / k
        result += term
        if matrix_inf_norm(term) <= local_tolerance:
            break
    else:
        raise ArithmeticError("The matrix-exponential series did not converge.")
    for _ in range(scaling):
        result = result * result
    return result


def exponential_skew_2x2(A: mp.matrix) -> mp.matrix:
    """Closed-form exponential of a 2x2 skew-symmetric matrix."""
    alpha = A[1, 0]
    c = mp.cos(alpha)
    s = mp.sin(alpha)
    return mp.matrix([[c, -s], [s, c]])


def euclidean_exponential(
    Y: mp.matrix,
    eta: mp.matrix,
    residual: mp.mpf | None = None,
) -> mp.matrix:
    """Riemannian exponential for the induced Euclidean metric.

    The block formula is evaluated using a scaled Taylor matrix exponential.
    The tolerance adapts to the residual. At the end, only the numerical
    orthogonality error is corrected through the polar factor of Y_raw.
    """
    A = Y.T * eta
    S = eta.T * eta
    block = block_matrix_2x2(A, -S, I2, A)
    tolerance = adaptive_exponential_tolerance(residual)
    exp_block = matrix_exponential_taylor(block, tolerance)
    # A is 2x2 skew-symmetric; therefore, so is -A.
    exp_minus_A = exponential_skew_2x2(-A)
    right = vertical_concatenate(exp_minus_A, mp.zeros(2, 2))
    Y_raw = horizontal_concatenate(Y, eta) * exp_block * right
    gram = Y_raw.T * Y_raw
    return Y_raw * inverse_square_root_spd_2x2(gram)


def inverse_square_root_spd_2x2(M: mp.matrix) -> mp.matrix:
    determinant = (
        M[0, 0] * M[1, 1]
        -
        M[0, 1] * M[1, 0]
    )

    if determinant <= 0:
        raise ArithmeticError("The matrix is not positive definite.")

    s = mp.sqrt(determinant)
    factor = mp.sqrt(trace(M) + 2 * s)

    return factor * mp.inverse(M + s * I2)


def polar_retraction(Y: mp.matrix, eta: mp.matrix) -> mp.matrix:
    M = I2 + eta.T * eta
    return (Y + eta) * inverse_square_root_spd_2x2(M)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def frame_error(Y: mp.matrix) -> mp.mpf:
    return frobenius_norm(Y - Y_STAR)


def estimate_orders(raw: list[dict]) -> None:
    for j, row in enumerate(raw):
        row["order"] = mp.nan

        if 1 <= j < len(raw) - 1:
            em1 = raw[j - 1]["error"]
            e0 = raw[j]["error"]
            ep1 = raw[j + 1]["error"]

            if em1 > 0 and e0 > 0 and ep1 > 0 and em1 != e0:
                row["order"] = mp.log(ep1 / e0) / mp.log(e0 / em1)


def run(
    update: Callable[[mp.matrix, mp.matrix], mp.matrix],
    method_name: str,
) -> list[dict[str, str]]:
    Y = mp.matrix(Y0_FRAME)
    raw: list[dict] = []

    for k in range(MAX_ITER + 1):
        X = field(Y)
        residual = frobenius_norm(X)
        error = frame_error(Y)
        defect = constraint_defect(Y)

        eta, M, _ = newton_direction(Y)

        raw.append({
            "method": method_name,
            "iteration": k,
            "Y": mp.matrix(Y),
            "residual": residual,
            "error": error,
            "defect": defect,
            "det_M": mp.det(M),
        })

        if residual <= TOL:
            break

        Y = update(Y, eta)
    else:
        raise RuntimeError(f"{method_name}: tolerance was not reached.")

    estimate_orders(raw)

    rows: list[dict[str, str]] = []

    for row in raw:
        Y = row["Y"]

        rows.append({
            "method": row["method"],
            "iteration": str(row["iteration"]),
            "y11": sci(Y[0, 0], 50),
            "y12": sci(Y[0, 1], 50),
            "y21": sci(Y[1, 0], 50),
            "y22": sci(Y[1, 1], 50),
            "y31": sci(Y[2, 0], 50),
            "y32": sci(Y[2, 1], 50),
            "residual": sci(row["residual"], 50),
            "frobenius_error": sci(row["error"], 50),
            "orthogonality_defect": sci(row["defect"], 30),
            "det_M": sci(row["det_M"], 50),
            "estimated_order": (
                ""
                if mp.isnan(row["order"])
                else sci(row["order"], 20)
            ),
        })

    return rows


def main() -> None:
    rows_exp = run(euclidean_exponential, "Newton + exponential")
    rows_ret = run(polar_retraction, "Newton + polar retraction")
    all_rows = rows_exp + rows_ret

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    exp_rows = {int(row["iteration"]): row for row in rows_exp}
    ret_rows = {int(row["iteration"]): row for row in rows_ret}
    ks = sorted(set(exp_rows) | set(ret_rows))

    def get(row, key, digits):
        if row is None or row.get(key, "") == "":
            return "-"
        return sci(mp.mpf(row[key]), digits)

    print("Applied problem 3: Brockett eigenframe on St(3,2)")
    print("A =")
    for i in range(3):
        print([sci(A_MATRIX[i, j], 14) for j in range(3)])
    print(f"tolerance={TOL}")
    print()
    print(
        f"{'k':>3}  "
        f"{'||X|| Exp':>24} {'order':>11}  "
        f"{'||X|| Polar':>24} {'order':>11}"
    )
    print("-" * 82)

    for k in ks:
        erow = exp_rows.get(k)
        rrow = ret_rows.get(k)

        print(
            f"{k:3d}  "
            f"{get(erow, 'residual', 14):>24} "
            f"{get(erow, 'estimated_order', 9):>11}  "
            f"{get(rrow, 'residual', 14):>24} "
            f"{get(rrow, 'estimated_order', 9):>11}"
        )

    print()
    print(
        "Iterations:",
        rows_exp[-1]["iteration"],
        "(exponential),",
        rows_ret[-1]["iteration"],
        "(polar retraction).",
    )
    print(
        "Maximum orthogonality defect:",
        max(mp.mpf(row["orthogonality_defect"]) for row in all_rows),
    )
    print("CSV:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
