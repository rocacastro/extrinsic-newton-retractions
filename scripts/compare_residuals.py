#!/usr/bin/env python3
"""Compare regenerated residual trajectories with committed reference data.

Timing columns are intentionally ignored because they depend on hardware,
operating-system scheduling, and system load.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse(value: str) -> Decimal | None:
    value = value.strip()
    return Decimal(value) if value else None


def close(a: Decimal, b: Decimal, relative: Decimal) -> bool:
    scale = max(Decimal(1), abs(a), abs(b))
    return abs(a - b) <= relative * scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=ROOT / "results_local",
        help="Directory produced by generate_article_tables.py",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=Decimal,
        default=Decimal("1e-40"),
    )
    args = parser.parse_args()

    reference = ROOT / "results" / "multiprecision"
    candidate = args.candidate_root / "multiprecision"
    files = sorted(path.name for path in reference.glob("table_*.csv"))
    if not files:
        raise RuntimeError("No committed reference trajectories were found")

    for filename in files:
        ref_path = reference / filename
        cand_path = candidate / filename
        if not cand_path.exists():
            raise RuntimeError(f"Missing candidate trajectory: {cand_path}")

        ref_rows = read_rows(ref_path)
        cand_rows = read_rows(cand_path)
        if len(ref_rows) != len(cand_rows):
            raise RuntimeError(f"{filename}: different number of rows")

        for ref, cand in zip(ref_rows, cand_rows):
            if ref["k"] != cand["k"]:
                raise RuntimeError(f"{filename}: iteration mismatch")
            for column in ("exp_residual", "ret_residual"):
                a, b = parse(ref[column]), parse(cand[column])
                if a is None or b is None:
                    if a != b:
                        raise RuntimeError(
                            f"{filename}, k={ref['k']}: missing-value mismatch"
                        )
                    continue
                if not close(a, b, args.relative_tolerance):
                    raise RuntimeError(
                        f"{filename}, k={ref['k']}, {column}: {a} != {b}"
                    )
        print(f"[OK] {filename}")

    print("Residual trajectories agree with the committed reference data.")


if __name__ == "__main__":
    main()
