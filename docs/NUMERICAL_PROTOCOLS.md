# Numerical protocols

## 1. Multiprecision trajectory protocol

Purpose: observe the asymptotic regime and compare the total accumulated cost up to the extreme stopping tolerance used in the article.

- arithmetic: `mpmath` arbitrary precision;
- decimal digits: 1200;
- tolerance: `1e-500`;
- repetitions: 3;
- method order: alternating between exponential and retraction;
- warm-up: one short warm-up per method at lower precision;
- reported trajectory time: median cumulative time at each iteration;
- final dispersion: median absolute deviation of the total time.

Each residual and cumulative time in a trajectory table belongs to the same protocol. If one method reaches the tolerance earlier, no artificial iterations are added.

## 2. Moderate-precision timing control

Purpose: verify whether the timing tendency persists at a less extreme precision.

- decimal digits: 50;
- tolerance: `1e-14`;
- warm-ups: 2;
- repetitions: 11;
- method order: alternating;
- reported statistic: median total time and median absolute deviation.

## 3. Interpretation

The timing results compare the implementations contained in this repository. They are not universal, hardware-independent complexity bounds. Absolute times may change with:

- processor and memory;
- operating system and Python build;
- background load;
- CPU frequency and power policy;
- mathematical-library configuration.

The intended reproducible objects are the algorithmic definitions, residual trajectories, stopping criteria, and statistical protocol. Timing ratios should be remeasured on the target machine.

## 4. `SL(2,R)` exponential

The Riemannian exponential for the Frobenius-induced metric is evaluated numerically from the extrinsic geodesic equation. The repository records:

- requested integration tolerance;
- Taylor-series order;
- a posteriori tail indicator;
- determinant defect;
- tangency defect;
- geodesic-equation residual;
- energy defect.

The a posteriori tail indicator is a numerical stopping criterion, not a formal interval-arithmetic certificate.
