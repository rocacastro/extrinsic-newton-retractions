# Committed reference results

The files in this directory are the reference numerical results associated with the English-language article.

- `multiprecision/`: complete residual and median cumulative-time trajectories;
- `moderate_precision/`: independent timing control at 50 digits;
- `diagnostics/`: numerical controls for the Riemannian exponential on `SL(2,R)`;
- `execution_environment.json`: reference execution environment;
- `metadata/`: archived metadata retained for provenance.

To avoid overwriting committed timing values, regenerate locally with:

```bash
python src/generate_article_tables.py --output-root results_local
```

Then compare deterministic residual trajectories with:

```bash
python scripts/compare_residuals.py --candidate-root results_local
```

Timing values are expected to change across machines.
