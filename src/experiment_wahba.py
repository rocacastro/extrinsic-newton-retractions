"""
Extrinsic Newton method for Wahba's problem with unit quaternions.

Ambient space:
    R^4.

Manifold:
    S^3 = {q in R^4 : ||q||=1}.

Quaternion convention:
    q=(q1,q2,q3,q4), with vector part (q1,q2,q3)
    and scalar part q4.

Synthetic data:
    A true attitude q_true is fixed, R(q_true) is constructed, and three
    noise-free vector observations are used:
        v_i = e_i,
        w_i = R(q_true)e_i,
    with positive weights that sum to one.

Tangent vector field:
    X(q) = (I-qq^T) K q,
where K is the symmetric Davenport matrix. Its zeros are the unit
 eigenvectors of K. The physical solution of interest is the eigenvector
 associated with the largest eigenvalue.

Newton direction:
    H(q) xi = -Z_q^T X(q),
    H(q)=Z_q^T (K-lambda(q)I) Z_q,
    lambda(q)=q^T K q,
    eta=Z_q xi.

Updates compared:
    1. The exact exponential on S^3.
    2. The normalization retraction.

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
MAX_ITER = 30
OUTPUT_CSV = Path("wahba_table.csv")

I4 = mp.eye(4)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dot(a: mp.matrix, b: mp.matrix) -> mp.mpf:
    return mp.fsum(a[i] * b[i] for i in range(len(a)))


def norm(a: mp.matrix) -> mp.mpf:
    return mp.sqrt(dot(a, a))


def normalize(a: mp.matrix) -> mp.matrix:
    return a / norm(a)


def sci(value: mp.mpf, digits: int = 18) -> str:
    if value == 0:
        return "0"
    return mp.nstr(value, digits)


def verification_tolerance() -> mp.mpf:
    """Tolerance consistent with the active mpmath precision."""
    return mp.power(10, -min(900, max(20, mp.mp.dps - 20)))


def small_threshold() -> mp.mpf:
    return mp.power(10, -min(1000, max(20, mp.mp.dps - 10)))


def skew(v: mp.matrix) -> mp.matrix:
    x, y, z = v[0], v[1], v[2]
    return mp.matrix([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])


# ---------------------------------------------------------------------------
# Quaternions and Davenport matrix
# ---------------------------------------------------------------------------

def quaternion_to_rotation(q: mp.matrix) -> mp.matrix:
    """
    Convention q=(v,s), with v in R^3 and scalar s:
        R(q)=(s^2-v^Tv)I+2vv^T+2s[v]_x.
    """
    v = mp.matrix([q[0], q[1], q[2]])
    s = q[3]
    vv = v * v.T

    return (
        (s * s - dot(v, v)) * mp.eye(3)
        +
        2 * vv
        +
        2 * s * skew(v)
    )


def build_wahba_data() -> tuple[
    mp.matrix,
    list[mp.matrix],
    list[mp.matrix],
    list[mp.mpf],
]:
    raw = mp.matrix([
        mp.mpf("0.30"),
        -mp.mpf("0.20"),
        mp.mpf("0.25"),
        mp.mpf("0.90"),
    ])
    q_true = normalize(raw)
    R_true = quaternion_to_rotation(q_true)

    reference_vectors = [
        mp.matrix([1, 0, 0]),
        mp.matrix([0, 1, 0]),
        mp.matrix([0, 0, 1]),
    ]

    weights = [
        mp.mpf("0.50"),
        mp.mpf("0.30"),
        mp.mpf("0.20"),
    ]

    observed_vectors = [
        R_true * v
        for v in reference_vectors
    ]

    return q_true, reference_vectors, observed_vectors, weights


def davenport_matrix(
    reference_vectors: list[mp.matrix],
    observed_vectors: list[mp.matrix],
    weights: list[mp.mpf],
) -> tuple[mp.matrix, mp.matrix]:
    """
    Profile matrix:
        B=sum_i a_i w_i v_i^T.

    With the convention q=(vector,scalar), define
        z=(B32-B23, B13-B31, B21-B12)^T,
        K=[[B+B^T-tr(B)I, z],[z^T,tr(B)]].
    """
    B = mp.zeros(3, 3)

    for a, v, w in zip(weights, reference_vectors, observed_vectors):
        B += a * (w * v.T)

    sigma = B[0, 0] + B[1, 1] + B[2, 2]
    S = B + B.T

    z = mp.matrix([
        B[2, 1] - B[1, 2],
        B[0, 2] - B[2, 0],
        B[1, 0] - B[0, 1],
    ])

    K = mp.matrix(4, 4)

    top_left = S - sigma * mp.eye(3)

    for i in range(3):
        for j in range(3):
            K[i, j] = top_left[i, j]
        K[i, 3] = z[i]
        K[3, i] = z[i]

    K[3, 3] = sigma

    return K, B


Q_TRUE, VECTORS_V, VECTORS_W, WEIGHTS = build_wahba_data()
K_MATRIX, PROFILE_B = davenport_matrix(VECTORS_V, VECTORS_W, WEIGHTS)


# ---------------------------------------------------------------------------
# Field, tangent basis, and Newton step
# ---------------------------------------------------------------------------

def tangent_basis(q: mp.matrix) -> mp.matrix:
    """
    Orthonormal basis Z_q in R^{4x3}, obtained by Gram--Schmidt.
    """
    basis: list[mp.matrix] = []
    threshold = small_threshold()

    for j in range(4):
        e = mp.matrix([1 if i == j else 0 for i in range(4)])
        h = e - dot(e, q) * q

        for z in basis:
            h -= dot(z, h) * z

        nh = norm(h)

        if nh > threshold:
            basis.append(h / nh)

        if len(basis) == 3:
            break

    if len(basis) != 3:
        raise ArithmeticError("Unable to construct T_q S^3.")

    Z = mp.matrix(4, 3)

    for j, z in enumerate(basis):
        for i in range(4):
            Z[i, j] = z[i]

    return Z


def field(q: mp.matrix) -> mp.matrix:
    lam = dot(q, K_MATRIX * q)
    return K_MATRIX * q - lam * q


def newton_direction(
    q: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    Z = tangent_basis(q)
    lam = dot(q, K_MATRIX * q)
    X = field(q)

    H = Z.T * (K_MATRIX - lam * I4) * Z
    rhs = -(Z.T * X)
    xi = mp.lu_solve(H, rhs)
    eta = Z * xi

    if abs(dot(q, eta)) > verification_tolerance():
        raise ArithmeticError("The Newton direction is not tangent.")

    if norm(H * xi + Z.T * X) > verification_tolerance():
        raise ArithmeticError("The reduced system was not solved correctly.")

    return eta, H, xi


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------

def exponential_update(q: mp.matrix, eta: mp.matrix) -> mp.matrix:
    r = norm(eta)

    if r == 0:
        return mp.matrix(q)

    return mp.cos(r) * q + (mp.sin(r) / r) * eta


def normalization_retraction(q: mp.matrix, eta: mp.matrix) -> mp.matrix:
    return normalize(q + eta)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def geodesic_error(q: mp.matrix) -> mp.mpf:
    """
    Distance on S^3, identifying q and -q as the same attitude.
    """
    cosine = abs(dot(q, Q_TRUE))
    cosine = min(mp.mpf("1"), max(mp.mpf("-1"), cosine))
    return mp.acos(cosine)


def initial_quaternion() -> mp.matrix:
    perturbation = mp.matrix([
        mp.mpf("0.18"),
        -mp.mpf("0.12"),
        mp.mpf("0.08"),
        -mp.mpf("0.05"),
    ])
    q0 = normalize(Q_TRUE + perturbation)

    if dot(q0, Q_TRUE) < 0:
        q0 = -q0

    return q0


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
    q = initial_quaternion()
    raw: list[dict] = []

    for k in range(MAX_ITER + 1):
        X = field(q)
        residual = norm(X)
        error = geodesic_error(q)
        norm_defect = abs(dot(q, q) - 1)

        eta, H, _ = newton_direction(q)

        raw.append({
            "method": method_name,
            "iteration": k,
            "q": mp.matrix(q),
            "residual": residual,
            "error": error,
            "norm_defect": norm_defect,
            "det_H": mp.det(H),
        })

        if residual <= TOL:
            break

        q = update(q, eta)

        if dot(q, Q_TRUE) < 0:
            q = -q
    else:
        raise RuntimeError(f"{method_name}: tolerance was not reached.")

    estimate_orders(raw)

    rows: list[dict[str, str]] = []

    for row in raw:
        q = row["q"]
        rows.append({
            "method": row["method"],
            "iteration": str(row["iteration"]),
            "q1": sci(q[0], 50),
            "q2": sci(q[1], 50),
            "q3": sci(q[2], 50),
            "q4": sci(q[3], 50),
            "residual": sci(row["residual"], 50),
            "geodesic_error": sci(row["error"], 50),
            "norm_defect": sci(row["norm_defect"], 30),
            "det_H": sci(row["det_H"], 50),
            "estimated_order": (
                ""
                if mp.isnan(row["order"])
                else sci(row["order"], 20)
            ),
        })

    return rows


def main() -> None:
    # Verify that q_true is the principal eigenvector.
    lambda_true = dot(Q_TRUE, K_MATRIX * Q_TRUE)
    eigen_residual = norm(K_MATRIX * Q_TRUE - lambda_true * Q_TRUE)

    if eigen_residual > verification_tolerance():
        raise ArithmeticError(
            "The K-matrix convention is not compatible with the true quaternion."
        )

    rows_exp = run(exponential_update, "Newton + exponential")
    rows_ret = run(normalization_retraction, "Newton + retraction")
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

    print("Applied problem 1: Wahba on S^3")
    print("Target eigenvalue =", sci(lambda_true, 20))
    print("q_true =", [sci(Q_TRUE[i], 14) for i in range(4)])
    print(f"tolerance={TOL}")
    print()
    print(
        f"{'k':>3}  "
        f"{'||X|| Exp':>24} {'order':>11}  "
        f"{'||X|| Retr':>24} {'order':>11}"
    )
    print("-" * 80)

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
        "(retraction).",
    )
    print(
        "Maximum norm defect:",
        max(mp.mpf(row["norm_defect"]) for row in all_rows),
    )
    print("CSV:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
