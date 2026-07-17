# GitHub upload guide

## 1. Create an empty repository

Create a new GitHub repository without automatically adding a README, license, or `.gitignore`, because those files are already included here.

Suggested repository name:

```text
extrinsic-newton-retractions
```

## 2. Inspect the release contents

The repository must contain the computational artifact and `supplement/`, but it must not contain the main article source or PDF.

Run:

```bash
python scripts/verify_repository.py
```

## 3. Initialize and push

From the extracted repository folder:

```bash
git init
git branch -M main
git add .
git commit -m "Initial code, data, and supplementary-material release"
git remote add origin https://github.com/OWNER/extrinsic-newton-retractions.git
git push -u origin main
```

Replace `OWNER` with the GitHub account or organization.

## 4. Complete public metadata before release

- choose licenses for software, data, documentation, and supplementary material;
- complete `CITATION.cff.template` and rename it to `CITATION.cff`;
- complete `.zenodo.json.template` and rename it to `.zenodo.json`;
- replace repository URL placeholders;
- confirm author order and ORCID identifiers.

## 5. Validate before tagging

```bash
python scripts/verify_repository.py
cd supplement
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
cd ..
python scripts/generate_checksums.py
```

Commit the updated control PDF and checksum manifest.

## 6. Create a release

```bash
git tag -a v1.1.0 -m "Version 1.1.0"
git push origin v1.1.0
```

Create the GitHub release from that tag. Connect the repository to Zenodo before the definitive release if a DOI is required.
