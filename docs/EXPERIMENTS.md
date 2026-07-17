# Experiments

The repository contains eight standalone experiments and one global generator. Each experiment computes the same extrinsic Newton direction for the two compared variants; only the update map changes.

## 1. Regular cylinder

**Script:** `src/experiment_leslie_regular_cylinder.py`

- Manifold: `x^2+z^2=1` in `R^3`.
- Field: `X=(-z psi, chi, x psi)`, with `psi=(x-z)+(x-z)^2` and `chi=y+y^2`.
- Regular zero: `(1/sqrt(2),0,1/sqrt(2))`.
- Exponential update: closed-form cylinder exponential.
- Retraction update: normalization of the circular component.
- Purpose: implementation of the reduced coordinate algorithm associated with Leslie San Martín's thesis.

## 2. Sphere `S^3`

**Script:** `src/experiment_sphere_s3_exp_retr.py`

- Manifold: the unit sphere in `R^4`.
- Field: `X(p)=e_1-<e_1,p>p`.
- Exponential update: exact sphere exponential.
- Retraction update: radial normalization.
- Special feature: the retraction reaches the zero in one update for the selected initial point.

## 3. Product torus

**Script:** `src/experiment_product_torus_exp_retr.py`

- Manifold: `S^1 x S^1` in `R^4`.
- Field:
  `X=f_1(y,w)(-y,x,0,0)+f_2(y,w)(0,0,-w,z)`.
- Regular zero: `(1,0,1,0)`.
- Exponential update: independent exact rotations on the two circle factors.
- Retraction update: componentwise normalization.
- Purpose: an explicit codimension-two regular level-set example with a reduced `2 x 2` Newton system.

## 4. Nonlinear `SL(2,R)` problem

**Script:** `src/experiment_sl2_riemannian_exponential_homogeneous_retraction.py`

- Manifold: `SL(2,R)` with the Frobenius-induced metric.
- Exponential update: numerical Riemannian exponential obtained by integrating the extrinsic geodesic equation.
- Retraction update: determinant normalization
  `R_A(eta)=det(A+eta)^(-1/2)(A+eta)`.
- Diagnostics: determinant defect, tangency defect, geodesic-equation residual, energy defect, Taylor-series order, and tail indicator.
- The group exponential is not used.

## 5. Nonlinear Stiefel field

**Script:** `src/experiment_st32_exponential_polar.py`

- Manifold: `St(3,2)` with the induced Euclidean metric.
- Field: `X(Y)=hat(g(Y))Y`, where `g(Y)` is a nonlinear polynomial system.
- Exponential update: induced-metric Stiefel exponential computed from a block matrix exponential.
- Retraction update: polar retraction.
- Purpose: comparison on a nonlinear tangent field over a matrix manifold.

## 6. Wahba problem

**Script:** `src/experiment_wahba.py`

- Manifold: unit quaternions `S^3`.
- Field: `X(q)=(I-qq^T)Kq`, where `K` is the Davenport matrix.
- Exponential update: exact sphere exponential.
- Retraction update: quaternion normalization.
- Purpose: synthetic attitude-estimation application.

## 7. Toroidal von Mises model

**Script:** `src/experiment_toroidal_von_mises.py`

- Manifold: `S^1 x S^1`.
- Field: tangent gradient of the negative log-density of a sine-type bivariate von Mises model.
- Exponential update: exact product-torus exponential.
- Retraction update: componentwise normalization.
- Purpose: localization of a local mode for synthetic periodic-angle data.

## 8. Brockett eigenproblem

**Script:** `src/experiment_brockett_stiefel.py`

- Manifold: `St(3,2)`.
- Field: `X_B(Y)=P_Y(AYN)` with symmetric `A` and diagonal `N`.
- Exponential update: induced-metric Stiefel exponential.
- Retraction update: polar retraction.
- Purpose: computation of an ordered orthonormal eigenvector frame.

## Global result generator

**Script:** `src/generate_article_tables.py`

This is not a ninth experiment. It imports the eight experiment modules and generates all official comparison data:

```bash
python src/generate_article_tables.py --output-root results_local
```

It produces:

- `results_local/multiprecision/`: complete residual and cumulative-time trajectories;
- `results_local/moderate_precision/`: total-time comparison at moderate precision;
- `results_local/diagnostics/`: `SL(2,R)` geodesic diagnostics;
- `results_local/execution_environment.json`: runtime metadata.
