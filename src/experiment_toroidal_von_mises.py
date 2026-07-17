"""
Extrinsic Newton method for a bivariate von Mises model on S^1 x S^1.

Application:
    Modeling pairs of protein dihedral angles (phi, psi).

Ambient space:
    R^4.

Manifold:
    S^1 x S^1 =
    {(x,y,z,w): x^2+y^2=1, z^2+w^2=1}.

Statistical model:
    Sine-type bivariate von Mises density
        pi(phi,psi) ∝ exp[
            kappa1 cos(phi-mu1)
            + kappa2 cos(psi-mu2)
            + lambda sin(phi-mu1) sin(psi-mu2)
        ].

Field:
    Gradient of the energy V=-log(pi), expressed in the tangent basis
    (e_phi,e_psi).

The following updates are compared:
    1. The exact exponential on the product torus.
    2. Componentwise normalization retraction on the two circles.

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
MAX_ITER = 30
OUTPUT_CSV = Path("toroidal_von_mises_table.csv")

KAPPA1 = mp.mpf("2.5")
KAPPA2 = mp.mpf("3.0")
LAMBDA = mp.mpf("1.2")
MU1 = -mp.mpf("1.1")
MU2 = mp.mpf("0.8")

THETA0 = MU1 + mp.mpf("0.5")
PHI0 = MU2 - mp.mpf("0.4")


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


def wrap_angle(angle: mp.mpf) -> mp.mpf:
    return mp.atan2(mp.sin(angle), mp.cos(angle))


# ---------------------------------------------------------------------------
# Product-torus geometry
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


def tangent_basis(p: mp.matrix) -> mp.matrix:
    x, y, z, w = p

    return mp.matrix([
        [-y, 0],
        [ x, 0],
        [ 0,-w],
        [ 0, z],
    ])


def tangent_projector(p: mp.matrix) -> mp.matrix:
    Z = tangent_basis(p)
    return Z * Z.T


# ---------------------------------------------------------------------------
# Sine-type von Mises field
# ---------------------------------------------------------------------------

def shifted_trigonometry(p: mp.matrix) -> tuple[
    mp.mpf,
    mp.mpf,
    mp.mpf,
    mp.mpf,
]:
    x, y, z, w = p

    c1 = mp.cos(MU1)
    s1 = mp.sin(MU1)
    c2 = mp.cos(MU2)
    s2 = mp.sin(MU2)

    sin_u = y * c1 - x * s1
    cos_u = x * c1 + y * s1
    sin_v = w * c2 - z * s2
    cos_v = z * c2 + w * s2

    return sin_u, cos_u, sin_v, cos_v


def coefficients(p: mp.matrix) -> tuple[mp.mpf, mp.mpf]:
    sin_u, cos_u, sin_v, cos_v = shifted_trigonometry(p)

    g1 = (
        KAPPA1 * sin_u
        -
        LAMBDA * cos_u * sin_v
    )

    g2 = (
        KAPPA2 * sin_v
        -
        LAMBDA * sin_u * cos_v
    )

    return g1, g2


def field(p: mp.matrix) -> mp.matrix:
    x, y, z, w = p
    g1, g2 = coefficients(p)

    return mp.matrix([
        -y * g1,
         x * g1,
        -w * g2,
         z * g2,
    ])


def derivatives_coefficients(
    p: mp.matrix,
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """
    Return the ambient gradients of g1 and g2 in R^4.
    """
    sin_u, cos_u, sin_v, cos_v = shifted_trigonometry(p)

    c1 = mp.cos(MU1)
    s1 = mp.sin(MU1)
    c2 = mp.cos(MU2)
    s2 = mp.sin(MU2)

    d_sin_u = [-s1, c1, 0, 0]
    d_cos_u = [ c1, s1, 0, 0]
    d_sin_v = [0, 0, -s2, c2]
    d_cos_v = [0, 0,  c2, s2]

    dg1 = []
    dg2 = []

    for j in range(4):
        dg1.append(
            KAPPA1 * d_sin_u[j]
            -
            LAMBDA * (
                d_cos_u[j] * sin_v
                +
                cos_u * d_sin_v[j]
            )
        )

        dg2.append(
            KAPPA2 * d_sin_v[j]
            -
            LAMBDA * (
                d_sin_u[j] * cos_v
                +
                sin_u * d_cos_v[j]
            )
        )

    return dg1, dg2


def ambient_jacobian(p: mp.matrix) -> mp.matrix:
    x, y, z, w = p
    g1, g2 = coefficients(p)
    dg1, dg2 = derivatives_coefficients(p)

    J = mp.matrix(4, 4)

    # X1=-y g1
    for j in range(4):
        J[0, j] = -y * dg1[j]
    J[0, 1] -= g1

    # X2=x g1
    for j in range(4):
        J[1, j] = x * dg1[j]
    J[1, 0] += g1

    # X3=-w g2
    for j in range(4):
        J[2, j] = -w * dg2[j]
    J[2, 3] -= g2

    # X4=z g2
    for j in range(4):
        J[3, j] = z * dg2[j]
    J[3, 2] += g2

    return J


def angular_hessian(p: mp.matrix) -> mp.matrix:
    sin_u, cos_u, sin_v, cos_v = shifted_trigonometry(p)

    return mp.matrix([
        [
            KAPPA1 * cos_u
            +
            LAMBDA * sin_u * sin_v,
            -LAMBDA * cos_u * cos_v,
        ],
        [
            -LAMBDA * cos_u * cos_v,
            KAPPA2 * cos_v
            +
            LAMBDA * sin_u * sin_v,
        ],
    ])


# ---------------------------------------------------------------------------
# Newton direction
# ---------------------------------------------------------------------------

def newton_direction(
    p: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix]:
    Z = tangent_basis(p)
    P = tangent_projector(p)
    J = ambient_jacobian(p)
    X = field(p)

    H = Z.T * P * J * Z
    rhs = -(Z.T * X)
    xi = mp.lu_solve(H, rhs)
    eta = Z * xi

    H_check = angular_hessian(p)

    h_difference = mp.matrix([
        H[0, 0] - H_check[0, 0],
        H[0, 1] - H_check[0, 1],
        H[1, 0] - H_check[1, 0],
        H[1, 1] - H_check[1, 1],
    ])

    if norm(h_difference) > verification_tolerance():
        raise ArithmeticError(
            "The extrinsic formulation does not agree with the angular Hessian."
        )

    if norm(H * xi + Z.T * X) > verification_tolerance():
        raise ArithmeticError("The Newton system was not solved.")

    return eta, H, xi


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------

def exponential_update(
    p: mp.matrix,
    eta: mp.matrix,
    xi: mp.matrix,
) -> mp.matrix:
    x, y, z, w = p
    alpha, beta = xi[0], xi[1]

    return mp.matrix([
        x * mp.cos(alpha) - y * mp.sin(alpha),
        y * mp.cos(alpha) + x * mp.sin(alpha),
        z * mp.cos(beta)  - w * mp.sin(beta),
        w * mp.cos(beta)  + z * mp.sin(beta),
    ])


def normalization_retraction(
    p: mp.matrix,
    eta: mp.matrix,
    xi: mp.matrix,
) -> mp.matrix:
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
# Execution
# ---------------------------------------------------------------------------

def geodesic_error(p: mp.matrix) -> mp.mpf:
    theta = mp.atan2(p[1], p[0])
    phi = mp.atan2(p[3], p[2])

    dtheta = wrap_angle(theta - MU1)
    dphi = wrap_angle(phi - MU2)

    return mp.sqrt(dtheta * dtheta + dphi * dphi)


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
        })

        if residual <= TOL:
            break

        p = update(p, eta, xi)
    else:
        raise RuntimeError(f"{method_name}: tolerance was not reached.")

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
            "estimated_order": (
                ""
                if mp.isnan(row["order"])
                else sci(row["order"], 20)
            ),
        })

    return rows


def main() -> None:
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

    print("Applied problem 2: toroidal von Mises model")
    print("Target mode: (mu1,mu2) =", MU1, MU2)
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
        "Maximum constraint defect:",
        max(mp.mpf(row["constraint_defect"]) for row in all_rows),
    )
    print("CSV:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
