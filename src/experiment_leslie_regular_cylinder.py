"""
Extrinsic Newton method on the cylinder x^2+z^2=1.

This program implements the coordinate algorithm from Leslie San Martín's
thesis:

1. Build the local tangent basis
       b1=(g_z,0,-g_x),  b2=(0,g_z,-g_y).
2. Compute the covariant derivative as the tangent projection of the
   ambient derivative.
3. Express both the field and the covariant derivative in the basis {b1,b2}.
4. Solve the 2x2 Newton system.
5. Update the point with either
       (a) the explicit cylinder exponential, or
       (b) a normalization retraction.

Angular coordinates are not used to compute the Newton direction.

Field:
    delta = x-z
    psi   = delta + delta^2
    chi   = y + y^2
    X(x,y,z)=(-z*psi, chi, x*psi).

Regular zero under study:
    p_*=(1/sqrt(2),0,1/sqrt(2)).

Tolerance:
    1e-500.

Dependency:
    mpmath
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
MAX_ITER = 50
OUTPUT_CSV = Path("leslie_regular_cylinder_table.csv")


# ---------------------------------------------------------------------------
# Elementary utilities
# ---------------------------------------------------------------------------

def dot(a: mp.matrix, b: mp.matrix) -> mp.mpf:
    return mp.fsum(a[i] * b[i] for i in range(len(a)))


def norm(a: mp.matrix) -> mp.mpf:
    return mp.sqrt(dot(a, a))


def determinant_2x2(A: mp.matrix) -> mp.mpf:
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def sci(x: mp.mpf, digits: int = 18) -> str:
    """Readable scientific notation without prior loss to double precision."""
    return mp.nstr(x, digits)


def verification_tolerance() -> mp.mpf:
    """Tolerance consistent with the active mpmath precision."""
    return mp.power(10, -min(900, max(20, mp.mp.dps - 20)))


def small_threshold() -> mp.mpf:
    return mp.power(10, -min(1000, max(20, mp.mp.dps - 10)))


# ---------------------------------------------------------------------------
# Cylinder geometry and vector field
# ---------------------------------------------------------------------------

def constraint(p: mp.matrix) -> mp.mpf:
    x, _, z = p
    return x * x + z * z - 1


def grad_F(p: mp.matrix) -> mp.matrix:
    x, _, z = p
    return mp.matrix([2 * x, mp.mpf("0"), 2 * z])


def field(p: mp.matrix) -> mp.matrix:
    """
    X=(-z*psi, chi, x*psi),
    psi=(x-z)+(x-z)^2, chi=y+y^2.
    """
    x, y, z = p
    delta = x - z
    psi = delta + delta * delta
    chi = y + y * y
    return mp.matrix([-z * psi, chi, x * psi])


def ambient_jacobian(p: mp.matrix) -> mp.matrix:
    """Ambient Jacobian D X~(p)."""
    x, y, z = p
    delta = x - z
    psi = delta + delta * delta
    psi_prime = 1 + 2 * delta

    return mp.matrix(
        [
            [-z * psi_prime, 0, -psi + z * psi_prime],
            [0, 1 + 2 * y, 0],
            [psi + x * psi_prime, 0, -x * psi_prime],
        ]
    )


# ---------------------------------------------------------------------------
# Leslie extrinsic step
# ---------------------------------------------------------------------------

def leslie_newton_direction(
    p: mp.matrix,
) -> tuple[mp.matrix, mp.matrix, mp.matrix, mp.matrix]:
    """
    Compute the Newton direction using exactly the coordinate construction
    from Leslie's thesis.

    Return:
        eta       Newton direction in R^3,
        H         2x2 matrix of the covariant derivative in the basis beta_p,
        X_beta    field coordinates in beta_p,
        xi        coordinates of eta in beta_p.
    """
    x, _, z = p
    gx, gy, gz = 2 * x, mp.mpf("0"), 2 * z

    if abs(gz) < small_threshold():
        raise ZeroDivisionError(
            "The Leslie chart requires g_z != 0. "
            "A different tangent chart must be selected."
        )

    # Unit normal and tangent projector.
    grad = mp.matrix([gx, gy, gz])
    normal = grad / norm(grad)
    projector = mp.eye(3) - normal * normal.T

    # Leslie local tangent basis, with the sign required so that
    # <grad g, b_i>=0.
    B = mp.matrix(
        [
            [gz, 0],
            [0, gz],
            [-gx, -gy],
        ]
    )

    # Extrinsic covariant derivative: G=P D X~.
    G = projector * ambient_jacobian(p)

    # Matrix of G|_{T_p C} in the basis beta_p.
    # For a tangent vector w, its first two ambient coordinates
    # recover its coordinates in beta_p after division by g_z.
    GB = G * B
    H = mp.matrix(
        [
            [GB[0, 0] / gz, GB[0, 1] / gz],
            [GB[1, 0] / gz, GB[1, 1] / gz],
        ]
    )

    # Field coordinates in beta_p.
    X = field(p)
    X_beta = mp.matrix([X[0] / gz, X[1] / gz])

    # Newton equation: H xi = -[X]_beta.
    xi = mp.lu_solve(H, -X_beta)
    eta = B * xi

    # Internal checks.
    tangency_field = dot(field(p), grad)
    tangency_eta = dot(eta, grad)
    scale = max(mp.mpf("1"), norm(field(p)), norm(eta), norm(grad))
    check_tol = verification_tolerance() * scale

    if abs(tangency_field) > check_tol:
        raise ArithmeticError("The computed field is not tangent.")
    if abs(tangency_eta) > check_tol:
        raise ArithmeticError("The Newton direction is not tangent.")

    return eta, H, X_beta, xi


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------

def exponential_update(p: mp.matrix, eta: mp.matrix) -> mp.matrix:
    """
    Explicit cylinder exponential in the ambient coordinates used in
    Leslie's thesis.

    If eta=(v_x,v_y,v_z), tangency gives
        v_x/z = -a,  v_z/x = a.
    """
    x, y, z = p
    vx, vy, vz = eta

    if abs(x) < small_threshold() or abs(z) < small_threshold():
        # Equivalent stable ambient formula used only as a safeguard.
        a = -z * vx + x * vz
        return mp.matrix(
            [
                x * mp.cos(a) - z * mp.sin(a),
                y + vy,
                z * mp.cos(a) + x * mp.sin(a),
            ]
        )

    tx = vx / z
    tz = vz / x

    return mp.matrix(
        [
            x * mp.cos(tx) + z * mp.sin(tx),
            y + vy,
            z * mp.cos(tz) + x * mp.sin(tz),
        ]
    )


def retraction_update(p: mp.matrix, eta: mp.matrix) -> mp.matrix:
    """
    Ambient retraction: add p+eta and normalize only the pair (x,z).
    """
    x, y, z = p
    vx, vy, vz = eta

    qx = x + vx
    qz = z + vz
    rho = mp.sqrt(qx * qx + qz * qz)

    if rho == 0:
        raise ZeroDivisionError("The retraction is not defined for this step.")

    return mp.matrix([qx / rho, y + vy, qz / rho])


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run(
    update: Callable[[mp.matrix, mp.matrix], mp.matrix],
    method_name: str,
) -> list[dict[str, str]]:
    p = mp.matrix(
        [
            mp.mpf("4") / 5,
            mp.mpf("1") / 2,
            mp.mpf("3") / 5,
        ]
    )

    rows: list[dict[str, str]] = []

    for k in range(MAX_ITER + 1):
        X = field(p)
        residual = norm(X)
        eta, H, _, _ = leslie_newton_direction(p)

        rows.append(
            {
                "method": method_name,
                "iteration": str(k),
                "x": sci(p[0], 40),
                "y": sci(p[1], 40),
                "z": sci(p[2], 40),
                "residual": sci(residual, 40),
                "det_H": sci(determinant_2x2(H), 40),
                "constraint_defect": sci(abs(constraint(p)), 20),
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
    all_rows = rows_exp + rows_ret

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("Regular field on the cylinder")
    print("p0 = (4/5, 1/2, 3/5)")
    print("p* = (1/sqrt(2), 0, 1/sqrt(2))")
    print(f"tolerance = {TOL}")
    print()

    selected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    exp_by_k = {int(row["iteration"]): row for row in rows_exp}
    ret_by_k = {int(row["iteration"]): row for row in rows_ret}

    print(f"{'k':>3}  {'Exp. residual':>24}  {'Ret. residual':>24}")
    print("-" * 58)
    for k in sorted(selected & exp_by_k.keys() & ret_by_k.keys()):
        print(
            f"{k:3d}  "
            f"{sci(mp.mpf(exp_by_k[k]['residual']), 14):>24}  "
            f"{sci(mp.mpf(ret_by_k[k]['residual']), 14):>24}"
        )

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
