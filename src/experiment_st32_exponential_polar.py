"""
Extrinsic Newton method on St(3,2) with the induced Euclidean metric.

Manifold:
    St(3,2) = {Y in R^{3x2}: Y^T Y = I_2}.

Nonlinear tangent vector field:
    X(Y) = hat(g(Y)) Y,
where g=(g1,g2,g3) is a polynomial system in the six entries of Y. Since
hat(g) is skew-symmetric, X(Y) is tangent.

The Newton direction is computed from the reduced 3x3 system in the global
basis
    B_i(Y)=K_i Y,
where K_i=hat(e_i), i=1,2,3.

The following updates are compared:
1. The Riemannian exponential for the induced Euclidean metric, evaluated
   using a 4x4 block matrix exponential.
2. The polar retraction
       R_Y(eta)=(Y+eta)(I+eta^T eta)^(-1/2),
   which requires only a 2x2 inverse square root.

Tolerance:
    1e-500.

Dependency:
    mpmath.
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
MAX_ITER = 40
OUTPUT_CSV = Path("st32_exponential_polar_table.csv")

I2 = mp.eye(2)
I4 = mp.eye(4)

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

KAPPA1 = mp.mpf("1.6119")
KAPPA2 = mp.mpf("1.2468")
KAPPA3 = mp.mpf("1.6423")


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


def hat(v: mp.matrix) -> mp.matrix:
    v1, v2, v3 = v[0], v[1], v[2]

    return mp.matrix([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0],
    ])


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
# Rotations and initialization
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


def initial_frame() -> mp.matrix:
    Q = (
        rotation_z(mp.mpf("0.10"))
        *
        rotation_y(mp.mpf("-0.10"))
        *
        rotation_x(mp.mpf("0.20"))
    )

    return Q[:, 0:2]


# ---------------------------------------------------------------------------
# Geometry of St(3,2)
# ---------------------------------------------------------------------------

def constraint_defect(Y: mp.matrix) -> mp.mpf:
    return frobenius_norm(Y.T * Y - I2)


def tangent_project(Y: mp.matrix, Z: mp.matrix) -> mp.matrix:
    """
    Orthogonal projection for the induced Euclidean metric:
        P_Y(Z)=Z-Y sym(Y^T Z).
    """
    return Z - Y * sym(Y.T * Z)


def tangent_basis(Y: mp.matrix) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    """
    Global basis B_i(Y)=K_i Y of T_Y St(3,2).
    """
    return tuple(K * Y for K in SO3_BASIS)


# ---------------------------------------------------------------------------
# Nonlinear field X(Y)=hat(g(Y))Y
# ---------------------------------------------------------------------------

def u_values(Y: mp.matrix) -> mp.matrix:
    y11, y12 = Y[0, 0], Y[0, 1]
    y21, y22 = Y[1, 0], Y[1, 1]
    y31, y32 = Y[2, 0], Y[2, 1]

    u1 = (
        y11
        + mp.mpf("0.4") * y22
        - mp.mpf("0.3") * y31
        + mp.mpf("0.2") * y12
    )

    u2 = (
        y21
        - mp.mpf("0.2") * y12
        + mp.mpf("0.5") * y32
        + mp.mpf("0.1") * y11
    )

    u3 = (
        y31
        + mp.mpf("0.3") * y22
        - mp.mpf("0.4") * y21
        + mp.mpf("0.2") * y32
    )

    return mp.matrix([u1, u2, u3])


def g_values(Y: mp.matrix) -> mp.matrix:
    u1, u2, u3 = u_values(Y)

    g1 = u1 + u2 * u2 - KAPPA1
    g2 = u2 + u1 * u3 - KAPPA2
    g3 = u3 + u1 * u1 + mp.mpf("0.5") * u2 * u3 - KAPPA3

    return mp.matrix([g1, g2, g3])


def derivative_u(H: mp.matrix) -> mp.matrix:
    h11, h12 = H[0, 0], H[0, 1]
    h21, h22 = H[1, 0], H[1, 1]
    h31, h32 = H[2, 0], H[2, 1]

    du1 = (
        h11
        + mp.mpf("0.4") * h22
        - mp.mpf("0.3") * h31
        + mp.mpf("0.2") * h12
    )

    du2 = (
        h21
        - mp.mpf("0.2") * h12
        + mp.mpf("0.5") * h32
        + mp.mpf("0.1") * h11
    )

    du3 = (
        h31
        + mp.mpf("0.3") * h22
        - mp.mpf("0.4") * h21
        + mp.mpf("0.2") * h32
    )

    return mp.matrix([du1, du2, du3])


def derivative_g(Y: mp.matrix, H: mp.matrix) -> mp.matrix:
    u1, u2, u3 = u_values(Y)
    du1, du2, du3 = derivative_u(H)

    dg1 = du1 + 2 * u2 * du2

    dg2 = (
        du2
        + du1 * u3
        + u1 * du3
    )

    dg3 = (
        du3
        + 2 * u1 * du1
        + mp.mpf("0.5") * (
            du2 * u3
            +
            u2 * du3
        )
    )

    return mp.matrix([dg1, dg2, dg3])


def field(Y: mp.matrix) -> mp.matrix:
    return hat(g_values(Y)) * Y


def ambient_derivative_field(Y: mp.matrix, H: mp.matrix) -> mp.matrix:
    return (
        hat(derivative_g(Y, H)) * Y
        +
        hat(g_values(Y)) * H
    )


def covariant_derivative_field(Y: mp.matrix, H: mp.matrix) -> mp.matrix:
    return tangent_project(
        Y,
        ambient_derivative_field(Y, H),
    )


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

    eta = mp.matrix(3, 2)

    for i in range(3):
        for j in range(2):
            eta[i, j] = 0

    for j in range(3):
        eta += xi[j] * basis[j]

    tangency_defect = frobenius_norm(
        Y.T * eta + eta.T * Y
    )

    newton_defect = frobenius_norm(
        covariant_derivative_field(Y, eta)
        +
        X
    )

    if tangency_defect > verification_tolerance():
        raise ArithmeticError(
            "The computed direction is not tangent."
        )

    if newton_defect > verification_tolerance():
        raise ArithmeticError(
            "The Newton equation was not solved correctly."
        )

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
    """
    For a 2x2 SPD matrix M:
        M^(1/2)=(M+sI)/sqrt(tr(M)+2s),
        s=sqrt(det(M)).
    Luego
        M^(-1/2)=sqrt(tr(M)+2s)(M+sI)^(-1).
    """
    determinant = (
        M[0, 0] * M[1, 1]
        -
        M[0, 1] * M[1, 0]
    )

    if determinant <= 0:
        raise ArithmeticError(
            "The matrix is not positive definite."
        )

    s = mp.sqrt(determinant)
    factor = mp.sqrt(trace(M) + 2 * s)

    return factor * mp.inverse(M + s * I2)


def polar_retraction(
    Y: mp.matrix,
    eta: mp.matrix,
) -> mp.matrix:
    M = I2 + eta.T * eta
    return (Y + eta) * inverse_square_root_spd_2x2(M)


# ---------------------------------------------------------------------------
# Execution and order estimation
# ---------------------------------------------------------------------------

def estimate_orders(raw: list[dict]) -> None:
    for j, row in enumerate(raw):
        row["order"] = mp.nan

        if 1 <= j < len(raw) - 1:
            rm1 = raw[j - 1]["residual"]
            r0 = raw[j]["residual"]
            rp1 = raw[j + 1]["residual"]

            if rm1 > 0 and r0 > 0 and rp1 > 0 and rm1 != r0:
                row["order"] = (
                    mp.log(rp1 / r0)
                    /
                    mp.log(r0 / rm1)
                )


def run(
    update: Callable[[mp.matrix, mp.matrix], mp.matrix],
    method_name: str,
) -> tuple[list[dict[str, str]], mp.matrix]:
    Y = initial_frame()
    raw: list[dict] = []

    for k in range(MAX_ITER + 1):
        X = field(Y)
        residual = frobenius_norm(X)
        defect = constraint_defect(Y)

        eta, M, xi = newton_direction(Y)

        raw.append({
            "method": method_name,
            "iteration": k,
            "Y": mp.matrix(Y),
            "residual": residual,
            "defect": defect,
            "det_M": mp.det(M),
            "step_norm": frobenius_norm(eta),
        })

        if residual <= TOL:
            break

        Y = update(Y, eta)
    else:
        raise RuntimeError(
            f"{method_name}: tolerance was not reached."
        )

    estimate_orders(raw)

    rows: list[dict[str, str]] = []

    for row in raw:
        Y = row["Y"]
        g = g_values(Y)

        rows.append({
            "method": row["method"],
            "iteration": str(row["iteration"]),
            "y11": sci(Y[0, 0], 50),
            "y12": sci(Y[0, 1], 50),
            "y21": sci(Y[1, 0], 50),
            "y22": sci(Y[1, 1], 50),
            "y31": sci(Y[2, 0], 50),
            "y32": sci(Y[2, 1], 50),
            "g1": sci(g[0], 50),
            "g2": sci(g[1], 50),
            "g3": sci(g[2], 50),
            "residual": sci(row["residual"], 50),
            "orthogonality_defect": sci(
                row["defect"],
                30,
            ),
            "det_M": sci(row["det_M"], 50),
            "step_norm": sci(row["step_norm"], 50),
            "estimated_order": (
                ""
                if mp.isnan(row["order"])
                else sci(row["order"], 20)
            ),
        })

    return rows, raw[-1]["Y"]


def main() -> None:
    rows_exp, root_exp = run(
        euclidean_exponential,
        "Newton + induced-metric exponential",
    )

    rows_polar, root_polar = run(
        polar_retraction,
        "Newton + polar retraction",
    )

    all_rows = rows_exp + rows_polar

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    exp_rows = {
        int(row["iteration"]): row
        for row in rows_exp
    }

    polar_rows = {
        int(row["iteration"]): row
        for row in rows_polar
    }

    ks = sorted(set(exp_rows) | set(polar_rows))

    def get(
        row: dict[str, str] | None,
        key: str,
        digits: int,
    ) -> str:
        if row is None or row.get(key, "") == "":
            return "-"

        return sci(mp.mpf(row[key]), digits)

    print("Extrinsic Newton method on St(3,2)")
    print("Field X(Y)=hat(g(Y))Y")
    print(f"tolerance={TOL}")
    print()

    print(
        f"{'k':>3}  "
        f"{'||X|| Exp':>24} {'order':>10}  "
        f"{'||X|| polar':>24} {'order':>10}"
    )
    print("-" * 82)

    for k in ks:
        erow = exp_rows.get(k)
        prow = polar_rows.get(k)

        print(
            f"{k:3d}  "
            f"{get(erow, 'residual', 14):>24} "
            f"{get(erow, 'estimated_order', 8):>10}  "
            f"{get(prow, 'residual', 14):>24} "
            f"{get(prow, 'estimated_order', 8):>10}"
        )

    print()
    print(
        "Iterations:",
        rows_exp[-1]["iteration"],
        "(exponential),",
        rows_polar[-1]["iteration"],
        "(polar).",
    )

    print(
        "Difference between roots:",
        sci(frobenius_norm(root_exp - root_polar), 20),
    )

    print(
        "Maximum orthogonality defect:",
        max(
            mp.mpf(row["orthogonality_defect"])
            for row in all_rows
        ),
    )

    print("Numerical root:")
    root = root_polar

    for i in range(3):
        print(
            "[",
            sci(root[i, 0], 20),
            ",",
            sci(root[i, 1], 20),
            "]",
        )

    print("det(M(root)) =", rows_polar[-1]["det_M"])
    print("CSV:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
