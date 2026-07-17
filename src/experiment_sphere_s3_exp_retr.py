"""
Extrinsic Newton method on the sphere S^3⊂R^4.

Field:
    X(p)=e_1-<e_1,p>p,   p∈S^3.

The Newton direction is computed using ambient matrices:
    P_p = I_4-pp^T,
    G_p = P_p D X~(p),
    H_p = Z_p^T G_p Z_p,
where the columns of Z_p form an orthonormal basis of T_pS^3.

Two updates of the same Newton step are compared:
    1. the exact exponential on S^3;
    2. the normalization retraction.

Tolerance: 1e-500.
Dependency: mpmath.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import mpmath as mp


mp.mp.dps = 1200
TOL = mp.mpf("1e-500")
MAX_ITER = 30
OUTPUT_CSV = Path("sphere_s3_exp_retr_table.csv")


def dot(a: mp.matrix, b: mp.matrix) -> mp.mpf:
    return mp.fsum(a[i] * b[i] for i in range(len(a)))


def norm(a: mp.matrix) -> mp.mpf:
    return mp.sqrt(dot(a, a))


def sci(x: mp.mpf, digits: int = 18) -> str:
    if x == 0:
        return "0"
    return mp.nstr(x, digits)


def verification_tolerance() -> mp.mpf:
    """Tolerance consistent with the active mpmath precision."""
    return mp.power(10, -min(900, max(20, mp.mp.dps - 20)))


def small_threshold() -> mp.mpf:
    return mp.power(10, -min(1000, max(20, mp.mp.dps - 10)))


def sphere_constraint(p: mp.matrix) -> mp.mpf:
    return dot(p, p) - 1


def field(p: mp.matrix) -> mp.matrix:
    x1 = p[0]
    e1 = mp.matrix([1, 0, 0, 0])
    return e1 - x1 * p


def ambient_jacobian(p: mp.matrix) -> mp.matrix:
    """
    D X~(p) = -p e_1^T - x_1 I_4.
    """
    x1 = p[0]
    e1 = mp.matrix([1, 0, 0, 0])
    return -(p * e1.T) - x1 * mp.eye(4)


def tangent_basis(p: mp.matrix) -> mp.matrix:
    """
    Deterministic orthonormal basis of T_pS^3 obtained by Gram--Schmidt
    from projections of the canonical basis.
    """
    basis: list[mp.matrix] = []
    threshold = small_threshold()

    for j in range(4):
        e = mp.matrix([1 if i == j else 0 for i in range(4)])
        v = e - dot(e, p) * p

        for q in basis:
            v -= dot(q, v) * q

        nv = norm(v)
        if nv > threshold:
            basis.append(v / nv)

        if len(basis) == 3:
            break

    if len(basis) != 3:
        raise ArithmeticError("Unable to construct a basis of T_pS^3.")

    Z = mp.matrix(4, 3)
    for j, q in enumerate(basis):
        for i in range(4):
            Z[i, j] = q[i]
    return Z


def newton_direction(
    p: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    """
    Solve the reduced Newton system:
        H xi = -Z^T X(p),
        eta = Z xi.
    """
    P = mp.eye(4) - p * p.T
    J = ambient_jacobian(p)
    G = P * J
    Z = tangent_basis(p)

    H = Z.T * G * Z
    rhs = -(Z.T * field(p))
    xi = mp.lu_solve(H, rhs)
    eta = Z * xi

    # Internal checks
    check = G * eta + field(p)
    if norm(check) > verification_tolerance():
        raise ArithmeticError("The Newton equation was not solved with sufficient accuracy.")
    if abs(dot(p, eta)) > verification_tolerance():
        raise ArithmeticError("The computed direction is not tangent.")

    return eta, H, Z


def exponential_update(p: mp.matrix, eta: mp.matrix) -> mp.matrix:
    r = norm(eta)
    if r == 0:
        return mp.matrix(p)
    return mp.cos(r) * p + (mp.sin(r) / r) * eta


def retraction_update(p: mp.matrix, eta: mp.matrix) -> mp.matrix:
    q = p + eta
    nq = norm(q)
    if nq == 0:
        raise ZeroDivisionError("The normalization retraction is not defined.")
    return q / nq


def run(
    update: Callable[[mp.matrix, mp.matrix], mp.matrix],
    method_name: str,
) -> list[dict[str, str]]:
    sqrt3 = mp.sqrt(3)
    p = mp.matrix(
        [
            sqrt3 / 2,
            1 / (2 * sqrt3),
            1 / (2 * sqrt3),
            1 / (2 * sqrt3),
        ]
    )

    rows: list[dict[str, str]] = []

    for k in range(MAX_ITER + 1):
        residual = norm(field(p))
        eta, H, _ = newton_direction(p)

        rows.append(
            {
                "method": method_name,
                "iteration": str(k),
                "x1": sci(p[0], 50),
                "x2": sci(p[1], 50),
                "x3": sci(p[2], 50),
                "x4": sci(p[3], 50),
                "residual": sci(residual, 50),
                "det_H": sci(mp.det(H), 50),
                "constraint_defect": sci(abs(sphere_constraint(p)), 20),
            }
        )

        if residual <= TOL:
            return rows

        p = update(p, eta)

    raise RuntimeError(
        f"{method_name}: tolerance was not reached within {MAX_ITER} iterations."
    )


def main() -> None:
    rows_exp = run(exponential_update, "Newton + exponential")
    rows_ret = run(retraction_update, "Newton + retraction")
    rows = rows_exp + rows_ret

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    exp_by_k = {int(row["iteration"]): row for row in rows_exp}
    ret_by_k = {int(row["iteration"]): row for row in rows_ret}
    all_k = sorted(set(exp_by_k) | set(ret_by_k))

    print("Sphere S^3 and field X(p)=e1-p1 p")
    print("p0=(sqrt(3)/2,1/(2sqrt(3)),1/(2sqrt(3)),1/(2sqrt(3)))")
    print("p*=(1,0,0,0)")
    print(f"tolerance={TOL}")
    print()
    print(f"{'k':>3}  {'Exp. residual':>26}  {'Ret. residual':>26}")
    print("-" * 62)

    for k in all_k:
        exp_val = exp_by_k.get(k, {}).get("residual", "")
        ret_val = ret_by_k.get(k, {}).get("residual", "")
        exp_txt = sci(mp.mpf(exp_val), 16) if exp_val else "-"
        ret_txt = sci(mp.mpf(ret_val), 16) if ret_val else "-"
        print(f"{k:3d}  {exp_txt:>26}  {ret_txt:>26}")

    print()
    print(
        "Iterations:",
        rows_exp[-1]["iteration"],
        "(exponential),",
        rows_ret[-1]["iteration"],
        "(retraction).",
    )
    print("Generated file:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
