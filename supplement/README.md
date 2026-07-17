# Supplementary material

This directory contains the English supplementary material associated with the computational repository.

## Files

- `supplementary_material.tex`: authoritative LaTeX source;
- `supplementary_material.pdf`: compiled control PDF.

The main article is not included in this repository.

## Compile

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
```

From the repository root, the same operation can be run with:

```bash
make supplement
```

The source refers to the English repository paths, including:

- `src/generate_article_tables.py`;
- `results/multiprecision/`;
- `results/moderate_precision/`;
- `results/diagnostics/`;
- `results/execution_environment.json`;
- `docs/REPRODUCIBILITY.md`;
- `docs/NUMERICAL_PROTOCOLS.md`;
- `scripts/verify_repository.py`;
- `checksums/SHA256SUMS.txt`.
