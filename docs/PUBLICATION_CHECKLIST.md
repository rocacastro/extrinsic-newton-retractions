# Public-release checklist

## Scientific and numerical content

- [ ] Confirm that the committed CSV files match the values reported in the definitive article and the committed supplementary material.
- [ ] Execute `python scripts/verify_repository.py` successfully.
- [ ] Run all eight experiments on the definitive reference machine.
- [ ] Save the complete execution log.
- [ ] Verify the runtime-environment metadata.
- [ ] Compile `supplement/supplementary_material.tex` and compare it with the committed PDF.
- [ ] Generate `checksums/SHA256SUMS.txt`.

## Authorship and metadata

- [ ] Confirm the definitive author order.
- [ ] Confirm affiliations and correspondence details.
- [ ] Confirm ORCID identifiers.
- [ ] Complete `CITATION.cff.template`.
- [ ] Complete `.zenodo.json.template`.

## Licensing

- [ ] Select a software license for `src/` and `scripts/`.
- [ ] Select an appropriate license for numerical data, documentation, and supplementary material.
- [ ] Replace `LICENSE_TO_CHOOSE.md` with final license information.

## Repository release

- [ ] Confirm that no main-article source or PDF is present.
- [ ] Create the GitHub repository.
- [ ] Push the complete repository tree.
- [ ] Create a tagged release, for example `v1.1.0`.
- [ ] Connect GitHub to Zenodo.
- [ ] Archive the tagged release.
- [ ] Add the real DOI to repository metadata and to the article only after the DOI exists.
