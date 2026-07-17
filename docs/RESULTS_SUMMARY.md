# Numerical results summary

## Multiprecision protocol

1200 decimal digits, tolerance `1e-500`, and three alternating repetitions. Times are final median cumulative times; MAD is the median absolute deviation of the total time.

| Experiment | Exp. iterations | Ret. iterations | Exp. time (s) | Ret. time (s) | Reduction |
|---|---:|---:|---:|---:|---:|
| Cylinder | 11 | 11 | 0.057418 | 0.034571 | 39.8% |
| Sphere `S^3` | 7 | 1 | 0.098664 | 0.014422 | 85.4% |
| Product torus | 11 | 11 | 0.110673 | 0.058621 | 47.0% |
| Nonlinear `SL(2,R)` | 10 | 10 | 4.586303 | 0.137583 | 97.0% |
| Nonlinear Stiefel field | 12 | 12 | 0.626350 | 0.271646 | 56.6% |
| Wahba on `S^3` | 7 | 7 | 0.098081 | 0.085713 | 12.6% |
| Toroidal von Mises model | 7 | 7 | 0.215074 | 0.188965 | 12.1% |
| Brockett on `St(3,2)` | 9 | 9 | 0.479302 | 0.256425 | 46.5% |

## Moderate-precision timing control

50 decimal digits, tolerance `1e-14`, two warm-ups, and eleven alternating repetitions.

| Experiment | Exp. iterations | Ret. iterations | Exp. median (s) | Ret. median (s) | Reduction |
|---|---:|---:|---:|---:|---:|
| Cylinder | 5 | 5 | 0.004506 | 0.003922 | 13.0% |
| Sphere `S^3` | 4 | 1 | 0.010709 | 0.002558 | 76.1% |
| Product torus | 6 | 6 | 0.009887 | 0.008421 | 14.8% |
| Nonlinear `SL(2,R)` | 5 | 5 | 0.172084 | 0.015709 | 90.9% |
| Nonlinear Stiefel field | 7 | 7 | 0.057641 | 0.033332 | 42.2% |
| Wahba on `S^3` | 3 | 3 | 0.008871 | 0.008168 | 7.9% |
| Toroidal von Mises model | 4 | 3 | 0.005742 | 0.004185 | 27.1% |
| Brockett on `St(3,2)` | 4 | 4 | 0.042311 | 0.027628 | 34.7% |

Absolute timings are machine-dependent. The committed values document the reference execution used in the associated article; local runs should preserve residual trajectories but may produce different timing values.
