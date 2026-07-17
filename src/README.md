# Python experiments

This directory contains eight standalone experiment modules and one global result generator.

| File | Experiment | Exponential update | Retraction update |
|---|---|---|---|
| `experiment_leslie_regular_cylinder.py` | Cylinder | Closed-form cylinder exponential | Normalization of the circular component |
| `experiment_sphere_s3_exp_retr.py` | Sphere `S^3` | Sphere exponential | Radial normalization |
| `experiment_product_torus_exp_retr.py` | Product torus | Componentwise circle exponential | Componentwise normalization |
| `experiment_sl2_riemannian_exponential_homogeneous_retraction.py` | `SL(2,R)` | Numerical Riemannian exponential for the induced Frobenius metric | Determinant normalization |
| `experiment_st32_exponential_polar.py` | Nonlinear Stiefel field | Induced-metric Stiefel exponential | Polar retraction |
| `experiment_wahba.py` | Wahba problem | Sphere exponential | Quaternion normalization |
| `experiment_toroidal_von_mises.py` | Toroidal von Mises model | Product-torus exponential | Componentwise normalization |
| `experiment_brockett_stiefel.py` | Brockett eigenproblem | Induced-metric Stiefel exponential | Polar retraction |

`generate_article_tables.py` loads these modules, executes both numerical protocols, writes all CSV files, and records the runtime environment.

The experiment modules are deliberately standalone so that the geometric objects, tangent bases, Newton directions, exponential maps, retractions, and diagnostics can be inspected directly.

A mathematical and computational overview of each script is provided in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
