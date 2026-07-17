# Repository structure

The main article is intentionally excluded. The English supplementary material and the complete computational artifact are included.

## `src/`

- one standalone Python module for each of the eight experiments;
- `generate_article_tables.py`, which regenerates all numerical outputs.

The experiment modules expose the geometric objects, tangent bases, Newton directions, exponential updates, retraction updates, and diagnostic routines used by the global generator.

## `results/`

- `multiprecision/`: full residual and cumulative-time trajectories at 1200 digits and tolerance `1e-500`;
- `moderate_precision/`: total-time control at 50 digits and tolerance `1e-14`;
- `diagnostics/`: numerical diagnostics for the Riemannian exponential on `SL(2,R)`;
- `metadata/`: archived runtime-environment metadata from the reference run;
- `execution_environment.json`: metadata from the latest committed reference execution.

## `supplement/`

- `supplementary_material.tex`: English LaTeX source;
- `supplementary_material.pdf`: compiled control PDF;
- `README.md`: compilation and consistency instructions.

The supplementary material refers only to paths that exist in this repository. The main article is not stored here.

## `logs/`

Stores the reference result-generation log and the final repository-validation log. Local logs can be stored here using filenames matching `local_*.log`, which are ignored by Git.

## `scripts/`

- `verify_repository.py`: validates file presence, absence of the main article, Python syntax, result tables, residual tolerances, time monotonicity, diagnostics, supplementary-material files, English filenames, and known Spanish remnants in public text;
- `compare_residuals.py`: compares regenerated residual trajectories with committed reference trajectories;
- `generate_checksums.py`: creates `checksums/SHA256SUMS.txt`.

## `docs/`

Contains public documentation for the experiments, numerical protocols, reproducibility, repository structure, numerical results, and release preparation.

## `checksums/`

Contains the SHA-256 integrity manifest for all committed files except the manifest itself and transient build artifacts.
