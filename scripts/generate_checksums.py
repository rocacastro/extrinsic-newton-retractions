#!/usr/bin/env python3
"""Generate a deterministic SHA-256 manifest for repository files."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checksums" / "SHA256SUMS.txt"
EXCLUDED_PARTS = {".git", "__pycache__", ".venv", "venv"}
EXCLUDED_SUFFIXES = {
    ".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls",
    ".out", ".run.xml", ".synctex.gz", ".toc",
}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if path == OUTPUT:
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    name = path.name
    if any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False
    return path.is_file()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(path for path in ROOT.rglob("*") if include(path))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
