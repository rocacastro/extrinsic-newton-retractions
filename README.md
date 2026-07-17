# Extrinsic Newton methods with retractions

This repository contains the complete computational artifact and the English supplementary material accompanying the article

> **Extrinsic Newton method with retractions for zeros of vector fields on regular level-set submanifolds**.

The main article is **not** included in this repository. The repository contains only:

- the eight Python experiments;
- the global table and timing generator;
- committed numerical results and diagnostics;
- reproducibility and verification utilities;
- the English supplementary material in LaTeX and PDF.

## Authors

- **Rodrigo Alberto Castro Marín** — corresponding author, [rodrigo.castrom@uv.cl](mailto:rodrigo.castrom@uv.cl)
- **Leslie San Martín** — [l.sanmartindonoso1@uandresbello.edu](mailto:l.sanmartindonoso1@uandresbello.edu)

Affiliation: Institute of Mathematics, Universidad de Valparaíso, Valparaíso, Chile.

## Mathematical setting

The manifold is represented as a regular level set

\[
\mathcal M=F^{-1}(0)\subset\mathbb R^N,
\]

and the Newton direction is computed through the reduced tangent system

\[
\bigl(Z_p^\top D\widetilde X(p)Z_p\bigr)\xi_p=-Z_p^\top X(p),
\qquad
\eta_p=Z_p\xi_p.
\]

The experiments compare updates based on the Riemannian exponential with updates based on first-order retractions.

## Repository structure

```text
src/                   Eight standalone experiments and the global result generator
results/
  multiprecision/      Full trajectories to tolerance 1e-500
  moderate_precision/  Timing control at tolerance 1e-14
  diagnostics/         Diagnostics for the SL(2,R) Riemannian exponential
  metadata/            Archived runtime-environment metadata
supplement/            English supplementary material in LaTeX and PDF
logs/                  Reference execution logs
scripts/               Verification, residual-comparison, and checksum utilities
docs/                  Experiment descriptions, protocols, reproducibility, and release notes
checksums/              SHA-256 manifest
```

A detailed map is available in [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md). The eight experiments are described in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Requirements

- Python 3.10 or newer
- `mpmath==1.3.0`

Install the numerical dependency with:

```bash
python -m pip install -r requirements.txt
```

Compiling the supplementary material additionally requires a standard LaTeX installation with `pdflatex` or `latexmk`.

## Regenerate all numerical results

From the repository root:

```bash
python src/generate_article_tables.py --output-root results_local
```

This command writes a local regeneration to `results_local/` and leaves the committed reference data unchanged. It regenerates:

- eight full multiprecision trajectory tables;
- the multiprecision summary;
- the moderate-precision timing summary;
- diagnostics for the Riemannian exponential on `SL(2,R)`;
- runtime-environment metadata.

Execution times depend on the hardware, operating system, Python build, system load, and power policy. Residual trajectories should be reproducible at the stated precision; timing values are expected to vary.

Compare regenerated residual trajectories with the committed reference data:

```bash
python scripts/compare_residuals.py --candidate-root results_local
```

Timing columns are intentionally excluded from this comparison because they are machine-dependent.

## Compile the supplementary material

From the repository root:

```bash
cd supplement
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

Alternatively:

```bash
make supplement
```

The committed `supplement/supplementary_material.pdf` is the control PDF generated from the committed LaTeX source.

## Verify repository consistency

```bash
python scripts/verify_repository.py
```

## Numerical protocols

The reference trajectory tables use 1200 decimal digits and the stopping tolerance `1e-500`. Each method is executed three times with alternating execution order. The tables report the median cumulative time at each iteration and the median absolute deviation of the final time.

A second timing protocol uses 50 decimal digits, tolerance `1e-14`, two warm-up runs, and eleven alternating repetitions.

See [docs/NUMERICAL_PROTOCOLS.md](docs/NUMERICAL_PROTOCOLS.md).

## Citation and license

The repository includes inactive templates for GitHub/Zenodo metadata:

- `CITATION.cff.template`
- `.zenodo.json.template`

The repository owner, release date, DOI, ORCID identifiers, and license must be completed and verified before public release. Do not rename the templates to active metadata files until all placeholders have been resolved.

No license has yet been selected. See [LICENSE_TO_CHOOSE.md](LICENSE_TO_CHOOSE.md).

## Language

All public-facing documentation, source-code comments and docstrings, console messages, result headers, metadata keys, filenames, and supplementary-material text in this package are in English.
