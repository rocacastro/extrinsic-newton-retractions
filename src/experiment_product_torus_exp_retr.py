"""
Extrinsic Newton method on the product torus S^1 x S^1 ⊂ R^4.

Manifold:
    M = {(x,y,z,w): x^2+y^2=1, z^2+w^2=1}.

Tangent vector field:
    f1(y,w) = y + (1/2)w + y^2 + (1/5)yw,
    f2(y,w) = -(1/3)y + w + w^2 + (1/4)yw,

    X(x,y,z,w)
      = f1(y,w)(-y,x,0,0)
        + f2(y,w)(0,0,-w,z).

Regular zero:
    p_*=(1,0,1,0).

The direction is computed through the extrinsic formulation:
    P_p = I - DF(p)^T(DF(p)DF(p)^T)^(-1)DF(p),
    H_p = Z_p^T P_p D X~(p) Z_p,
    H_p xi = -Z_p^T X(p),
    eta = Z_p xi.

The following updates are compared:
1. Newton plus the exact exponential on the product torus.
2. Newton plus componentwise normalization retraction.

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

THETA0 = mp.mpf("0.45")
PHI0 = mp.mpf("-0.35")

OUTPUT_CSV = Path("product_torus_exp_retr_table.csv")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dot(a: mp.matrix, b: mp.matrix) -> mp.mpf:
    return mp.fsum(a[i] * b[i] for i in range(len(a)))


def norm(a: mp.matrix) -> mp.mpf:
    return mp.sqrt(dot(a, a))


def sci(value: mp.mpf, digits: int = 18) -> str:
    if value == 0:
        return "0"
    return mp.nstr(value, digits)


def verification_tolerance() -> mp.mpf:
    """Tolerance consistent with the active mpmath precision."""
    return mp.power(10, -min(900, max(20, mp.mp.dps - 20)))


def geodesic_error(p: mp.matrix) -> mp.mpf:
    """
    Local geodesic distance to the zero p_*=(1,0,1,0).
    """
    theta = mp.atan2(p[1], p[0])
    phi = mp.atan2(p[3], p[2])
    return mp.sqrt(theta * theta + phi * phi)


# ---------------------------------------------------------------------------
# Product torus and extrinsic geometry
# ---------------------------------------------------------------------------

def point_from_angles(theta: mp.mpf, phi: mp.mpf) -> mp.matrix:
    return mp.matrix([
        mp.cos(theta),
        mp.sin(theta),
        mp.cos(phi),
        mp.sin(phi),
    ])


def constraints(p: mp.matrix) -> mp.matrix:
    x, y, z, w = p
    return mp.matrix([
        x * x + y * y - 1,
        z * z + w * w - 1,
    ])


def DF(p: mp.matrix) -> mp.matrix:
    x, y, z, w = p
    return mp.matrix([
        [2 * x, 2 * y, 0, 0],
        [0, 0, 2 * z, 2 * w],
    ])


def tangent_basis(p: mp.matrix) -> mp.matrix:
    """
    Orthonormal basis Z=[e_theta,e_phi] of T_pM.
    """
    x, y, z, w = p
    return mp.matrix([
        [-y, 0],
        [ x, 0],
        [ 0,-w],
        [ 0, z],
    ])


def tangent_projector(p: mp.matrix) -> mp.matrix:
    A = DF(p)
    return mp.eye(4) - A.T * mp.inverse(A * A.T) * A


# ---------------------------------------------------------------------------
# Vector field
# ---------------------------------------------------------------------------

A = mp.mpf("0.5")
B = -mp.mpf(1) / 3
C = mp.mpf("0.2")
D = mp.mpf("0.25")


def coefficients(p: mp.matrix) -> tuple[mp.mpf, mp.mpf]:
    _, y, _, w = p

    f1 = y + A * w + y * y + C * y * w
    f2 = B * y + w + w * w + D * y * w

    return f1, f2


def field(p: mp.matrix) -> mp.matrix:
    x, y, z, w = p
    f1, f2 = coefficients(p)

    return mp.matrix([
        -y * f1,
         x * f1,
        -w * f2,
         z * f2,
    ])


def ambient_jacobian(p: mp.matrix) -> mp.matrix:
    """
    Jacobian of the ambient extension:
        X~=(-y f1, x f1, -w f2, z f2).
    """
    x, y, z, w = p
    f1, f2 = coefficients(p)

    f1_y = 1 + 2 * y + C * w
    f1_w = A + C * y

    f2_y = B + D * w
    f2_w = 1 + 2 * w + D * y

    return mp.matrix([
        [0, -f1 - y * f1_y, 0, -y * f1_w],
        [f1, x * f1_y,      0,  x * f1_w],
        [0, -w * f2_y,      0, -f2 - w * f2_w],
        [0,  z * f2_y,     f2,  z * f2_w],
    ])


def analytic_reduced_matrix(p: mp.matrix) -> mp.matrix:
    """
    Equivalent reduced matrix obtained in angular coordinates.
    It is used only to verify the extrinsic implementation.
    """
    x, y, z, w = p

    return mp.matrix([
        [
            x * (1 + 2 * y + C * w),
            z * (A + C * y),
        ],
        [
            x * (B + D * w),
            z * (1 + 2 * w + D * y),
        ],
    ])


# ---------------------------------------------------------------------------
# Newton direction
# ---------------------------------------------------------------------------

def newton_direction(
    p: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    """
    Solve the reduced extrinsic system:
        H xi = -Z^T X,
        eta = Z xi.
    """
    P = tangent_projector(p)
    Z = tangent_basis(p)
    J = ambient_jacobian(p)
    X = field(p)

    H = Z.T * P * J * Z
    rhs = -(Z.T * X)
    xi = mp.lu_solve(H, rhs)
    eta = Z * xi

    # Internal checks.
    H_exact = analytic_reduced_matrix(p)

    if norm(mp.matrix([
        H[0, 0] - H_exact[0, 0],
        H[0, 1] - H_exact[0, 1],
        H[1, 0] - H_exact[1, 0],
        H[1, 1] - H_exact[1, 1],
    ])) > verification_tolerance():
        raise ArithmeticError(
            "The extrinsic matrix does not agree with the angular-coordinate matrix."
        )

    if norm(H * xi + Z.T * X) > verification_tolerance():
        raise ArithmeticError(
            "The reduced Newton system was not solved correctly."
        )

    if norm(DF(p) * eta) > verification_tolerance():
        raise ArithmeticError(
            "The computed direction does not belong to the tangent space."
        )

    return eta, H, xi


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------

def exponential_update(
    p: mp.matrix,
    eta: mp.matrix,
    xi: mp.matrix,
) -> mp.matrix:
    """
    Exact exponential on the product S^1 x S^1.
    """
    x, y, z, w = p
    alpha, beta = xi[0], xi[1]

    return mp.matrix([
        x * mp.cos(alpha) - y * mp.sin(alpha),
        y * mp.cos(alpha) + x * mp.sin(alpha),
        z * mp.cos(beta)  - w * mp.sin(beta),
        w * mp.cos(beta)  + z * mp.sin(beta),
    ])


def retraction_update(
    p: mp.matrix,
    eta: mp.matrix,
    xi: mp.matrix,
) -> mp.matrix:
    """
    Retraction by separate normalization of each pair.
    """
    x, y, z, w = p
    alpha, beta = xi[0], xi[1]

    rho1 = mp.sqrt(1 + alpha * alpha)
    rho2 = mp.sqrt(1 + beta * beta)

    return mp.matrix([
        (x - alpha * y) / rho1,
        (y + alpha * x) / rho1,
        (z - beta * w) / rho2,
        (w + beta * z) / rho2,
    ])


# ---------------------------------------------------------------------------
# Execution and order estimation
# ---------------------------------------------------------------------------

def estimate_orders(raw: list[dict]) -> None:
    for j, row in enumerate(raw):
        row["order"] = mp.nan
        row["Q2"] = mp.nan

        if j < len(raw) - 1:
            e0 = raw[j]["error"]
            e1 = raw[j + 1]["error"]
            if e0 > 0:
                row["Q2"] = e1 / (e0 * e0)

        if 1 <= j < len(raw) - 1:
            em1 = raw[j - 1]["error"]
            e0 = raw[j]["error"]
            ep1 = raw[j + 1]["error"]

            if em1 > 0 and e0 > 0 and ep1 > 0 and em1 != e0:
                row["order"] = (
                    mp.log(ep1 / e0)
                    /
                    mp.log(e0 / em1)
                )


def run(
    update: Callable[[mp.matrix, mp.matrix, mp.matrix], mp.matrix],
    method_name: str,
) -> list[dict[str, str]]:
    p = point_from_angles(THETA0, PHI0)
    raw: list[dict] = []

    for k in range(MAX_ITER + 1):
        X = field(p)
        residual = norm(X)
        error = geodesic_error(p)
        defect = norm(constraints(p))

        eta, H, xi = newton_direction(p)

        raw.append({
            "method": method_name,
            "iteration": k,
            "p": mp.matrix(p),
            "residual": residual,
            "error": error,
            "defect": defect,
            "det_H": mp.det(H),
            "alpha": xi[0],
            "beta": xi[1],
        })

        if residual <= TOL:
            break

        p = update(p, eta, xi)
    else:
        raise RuntimeError(
            f"{method_name}: tolerance was not reached."
        )

    estimate_orders(raw)

    rows: list[dict[str, str]] = []

    for row in raw:
        p = row["p"]

        rows.append({
            "method": row["method"],
            "iteration": str(row["iteration"]),
            "x": sci(p[0], 50),
            "y": sci(p[1], 50),
            "z": sci(p[2], 50),
            "w": sci(p[3], 50),
            "residual": sci(row["residual"], 50),
            "geodesic_error": sci(row["error"], 50),
            "constraint_defect": sci(row["defect"], 30),
            "det_H": sci(row["det_H"], 50),
            "alpha": sci(row["alpha"], 50),
            "beta": sci(row["beta"], 50),
            "estimated_order": (
                ""
                if mp.isnan(row["order"])
                else sci(row["order"], 20)
            ),
            "Q2": (
                ""
                if mp.isnan(row["Q2"])
                else sci(row["Q2"], 20)
            ),
        })

    return rows


def main() -> None:
    rows_exp = run(exponential_update, "Newton + exponential")
    rows_ret = run(retraction_update, "Newton + retraction")
    all_rows = rows_exp + rows_ret

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    exp = {int(row["iteration"]): row for row in rows_exp}
    ret = {int(row["iteration"]): row for row in rows_ret}
    ks = sorted(set(exp) | set(ret))

    def get(row: dict[str, str] | None, key: str, digits: int) -> str:
        if row is None or row.get(key, "") == "":
            return "-"
        return sci(mp.mpf(row[key]), digits)

    print("Product torus S^1 x S^1")
    print("p*=(1,0,1,0)")
    print(f"theta0={THETA0}, phi0={PHI0}")
    print(f"tolerance={TOL}")
    print()

    print(
        f"{'k':>3}  "
        f"{'||X|| Exp':>22} {'order':>10} {'Q2':>12}  "
        f"{'||X|| Retr':>22} {'order':>10} {'Q2':>12}"
    )
    print("-" * 96)

    for k in ks:
        e = exp.get(k)
        r = ret.get(k)

        print(
            f"{k:3d}  "
            f"{get(e,'residual',13):>22} "
            f"{get(e,'estimated_order',8):>10} "
            f"{get(e,'Q2',8):>12}  "
            f"{get(r,'residual',13):>22} "
            f"{get(r,'estimated_order',8):>10} "
            f"{get(r,'Q2',8):>12}"
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
        "Maximum constraint violation:",
        max(mp.mpf(row["constraint_defect"]) for row in all_rows),
    )

    print("CSV file:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
