# Reproducibility guide

## Create an isolated Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Regenerate all data

```bash
python src/generate_article_tables.py --output-root results_local 2>&1 | tee logs/local_generation.log
```

On Windows Command Prompt, the direct command is:

```bat
python src\generate_article_tables.py --output-root results_local
```

The full run includes arbitrary-precision calculations and the numerical Riemannian exponential on `SL(2,R)`. Runtime depends strongly on the machine.

## Compare with committed reference residuals

```bash
python scripts/compare_residuals.py --candidate-root results_local
```

The committed CSV files document one controlled execution. Residuals should agree to the precision implied by the code. Timing values need not be identical.

Record a new execution environment in `results_local/execution_environment.json` and preserve the generated log whenever new timing results are reported.

## Compile the supplementary material

```bash
cd supplement
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
cd ..
```

The supplementary tables are based on the committed reference CSV files. When reference results are intentionally replaced, update the supplementary source and control PDF in the same commit.

## Verify the repository

```bash
python scripts/verify_repository.py
```

## Regenerate checksums

```bash
python scripts/generate_checksums.py
```

## Release through GitHub and Zenodo

Before creating a release:

1. confirm author order and ORCID identifiers;
2. select licenses for software, numerical data, documentation, and supplementary material;
3. complete `CITATION.cff.template` and rename it to `CITATION.cff`;
4. complete `.zenodo.json.template` and rename it to `.zenodo.json`;
5. run the verification and checksum scripts;
6. confirm that the main article is absent;
7. create a versioned GitHub release;
8. archive the release in Zenodo;
9. insert the DOI into the article only after the DOI exists.
