# Contributing

This repository is primarily a computational reproducibility artifact for an academic article.

Before opening a pull request:

1. create a separate branch;
2. explain whether the change affects mathematics, numerical results, documentation, or only formatting;
3. run `python scripts/verify_repository.py`;
4. regenerate checksums with `python scripts/generate_checksums.py`;
5. do not replace numerical values manually;
6. when results change, include the command, runtime-environment metadata, and execution log used to regenerate them;
7. keep all public text, source-code comments, docstrings, console output, filenames, metadata keys, and table headers in English.

Mathematical changes should preserve the notation

- `M = F^{-1}(0)`;
- `D\widetilde X(p)(\eta)`;
- `\nabla_\eta X(p)`.

Bibliographic claims must be verified against primary sources.
