#!/usr/bin/env python3
"""Check the computational repository for internal consistency."""

from __future__ import annotations

import csv
import json
import py_compile
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_FILES = {
    "cylinder": "table_cylinder.csv",
    "sphere_s3": "table_sphere_s3.csv",
    "product_torus": "table_product_torus.csv",
    "sl2_nonlinear": "table_sl2_nonlinear.csv",
    "stiefel_nonlinear": "table_stiefel_nonlinear.csv",
    "wahba": "table_wahba.csv",
    "von_mises": "table_von_mises.csv",
    "brockett": "table_brockett.csv",
}

REQUIRED = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "src" / "generate_article_tables.py",
    ROOT / "results" / "multiprecision" / "multiprecision_summary.csv",
    ROOT / "results" / "moderate_precision" / "moderate_precision_timing_summary.csv",
    ROOT / "results" / "diagnostics" / "sl2_geodesic_diagnostics.csv",
    ROOT / "results" / "execution_environment.json",
    ROOT / "supplement" / "supplementary_material.tex",
    ROOT / "supplement" / "supplementary_material.pdf",
    ROOT / "supplement" / "README.md",
]

FORBIDDEN_TOP_LEVEL = [
    ROOT / "paper",
    ROOT / "manuscript",
]

ALLOWED_DOCUMENTS = {
    Path("supplement/supplementary_material.tex"),
    Path("supplement/supplementary_material.pdf"),
}
FORBIDDEN_DOCUMENT_SUFFIXES = {".tex", ".pdf", ".bib"}

# Known Spanish remnants from the original partial repository. Proper names are
# intentionally excluded from this list.
FORBIDDEN_SPANISH_MARKERS = [
    "generar tablas",
    "resultados escritos",
    "tiempo acumulado",
    "residuo exponencial",
    "residuo retracción",
    "residuo retraccion",
    "experimento aplicado",
    "campo acoplado",
    "campo desacoplado",
    "archivo generado",
    "tolerancia no alcanzada",
    "dirección de newton",
    "direccion de newton",
    "método de newton",
    "metodo de newton",
    "exponencial riemanniana y retracción",
    "exponencial riemanniana y retraccion",
]

EXPECTED_TRAJECTORY_HEADERS = [
    "k",
    "exp_residual",
    "ret_residual",
    "exp_cumulative_time_s",
    "ret_cumulative_time_s",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def check_required_files() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        fail("Missing required files: " + ", ".join(missing))

    forbidden = [str(path.relative_to(ROOT)) for path in FORBIDDEN_TOP_LEVEL if path.exists()]
    if forbidden:
        fail("Excluded main-article directories are present: " + ", ".join(forbidden))


def check_python_sources() -> None:
    sources = sorted((ROOT / "src").glob("*.py"))
    if len(sources) != 9:
        fail(f"Expected 9 Python source files in src/, found {len(sources)}")
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        for source in sources:
            py_compile.compile(
                str(source),
                cfile=str(temp / f"{source.stem}.pyc"),
                doraise=True,
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_headers(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        return next(reader)


def decimal_or_none(value: str) -> Decimal | None:
    value = value.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise RuntimeError(f"Invalid decimal value {value!r}") from exc


def check_trajectory(path: Path) -> None:
    headers = csv_headers(path)
    if headers != EXPECTED_TRAJECTORY_HEADERS:
        fail(f"Unexpected headers in {path.relative_to(ROOT)}: {headers}")

    rows = read_csv(path)
    if not rows:
        fail(f"Empty trajectory file: {path.relative_to(ROOT)}")

    expected_k = list(range(len(rows)))
    observed_k = [int(row["k"]) for row in rows]
    if observed_k != expected_k:
        fail(f"Nonconsecutive iteration indices in {path.name}")

    for time_column in ("exp_cumulative_time_s", "ret_cumulative_time_s"):
        values = [decimal_or_none(row[time_column]) for row in rows]
        values = [value for value in values if value is not None]
        if any(b < a for a, b in zip(values, values[1:])):
            fail(f"Nonmonotone cumulative times in {path.name}: {time_column}")

    for residual_column in ("exp_residual", "ret_residual"):
        values = [decimal_or_none(row[residual_column]) for row in rows]
        values = [value for value in values if value is not None]
        if not values or any(value < 0 for value in values):
            fail(f"Invalid residuals in {path.name}: {residual_column}")
        if values[-1] > Decimal("1e-500"):
            fail(f"Final residual exceeds 1e-500 in {path.name}: {residual_column}")


def check_results() -> None:
    multi_dir = ROOT / "results" / "multiprecision"
    summary_path = multi_dir / "multiprecision_summary.csv"
    summary = read_csv(summary_path)
    if len(summary) != 8:
        fail(f"Expected 8 multiprecision summary rows, found {len(summary)}")

    summary_files = {row["file"] for row in summary}
    expected_files = set(EXPERIMENT_FILES.values())
    if summary_files != expected_files:
        fail(
            "The multiprecision summary does not reference the expected "
            f"trajectory files: expected={sorted(expected_files)}, "
            f"found={sorted(summary_files)}"
        )

    for filename in sorted(expected_files):
        check_trajectory(multi_dir / filename)

    moderate_path = (
        ROOT / "results" / "moderate_precision"
        / "moderate_precision_timing_summary.csv"
    )
    moderate = read_csv(moderate_path)
    if len(moderate) != 8:
        fail(f"Expected 8 moderate-precision rows, found {len(moderate)}")

    diagnostics_path = (
        ROOT / "results" / "diagnostics" / "sl2_geodesic_diagnostics.csv"
    )
    diagnostics = read_csv(diagnostics_path)
    if not diagnostics:
        fail("SL(2,R) diagnostics are empty")

    metadata_path = ROOT / "results" / "execution_environment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mpmath") != "1.3.0":
        fail("Runtime-environment metadata must record mpmath 1.3.0")
    if "moderate_precision_timings" not in metadata:
        fail("Runtime-environment metadata lacks moderate-precision settings")


def check_document_scope() -> None:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in FORBIDDEN_DOCUMENT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative not in ALLOWED_DOCUMENTS:
            offenders.append(relative.as_posix())
    if offenders:
        fail(
            "Document files outside the approved supplementary-material scope are present: "
            + ", ".join(sorted(offenders))
        )

    pdf_path = ROOT / "supplement" / "supplementary_material.pdf"
    if not pdf_path.read_bytes().startswith(b"%PDF"):
        fail("The supplementary control PDF is not a valid PDF file")

    tex_path = ROOT / "supplement" / "supplementary_material.tex"
    tex = tex_path.read_text(encoding="utf-8")
    required_paths = [
        "src/generate_article_tables.py",
        "results/multiprecision",
        "results/moderate_precision",
        "results/diagnostics",
        "results/execution_environment.json",
        "docs/REPRODUCIBILITY.md",
        "docs/NUMERICAL_PROTOCOLS.md",
        "scripts/verify_repository.py",
        "checksums/SHA256SUMS.txt",
    ]
    missing_references = [value for value in required_paths if value not in tex]
    if missing_references:
        fail(
            "The supplementary source lacks expected English repository references: "
            + ", ".join(missing_references)
        )


def check_english_public_text() -> None:
    offenders: list[str] = []
    text_suffixes = {".md", ".py", ".csv", ".json", ".txt", ".template", ".tex", ""}
    excluded_parts = {"checksums", "__pycache__"}
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix.lower() not in text_suffixes and path.name not in {
            "Makefile", ".gitignore"
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        matches = [marker for marker in FORBIDDEN_SPANISH_MARKERS if marker in text]
        if matches:
            offenders.append(
                f"{path.relative_to(ROOT).as_posix()}: {', '.join(matches)}"
            )
    if offenders:
        fail("Spanish remnants found in public text: " + "; ".join(offenders))


def check_english_filenames() -> None:
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        try:
            relative.encode("ascii")
        except UnicodeEncodeError:
            offenders.append(relative)
            continue
        if " " in relative:
            offenders.append(relative)
    if offenders:
        fail(
            "Repository filenames must use ASCII characters and no spaces: "
            + ", ".join(offenders)
        )


def main() -> int:
    checks = [
        ("required files and excluded main-article directories", check_required_files),
        ("document scope and supplementary references", check_document_scope),
        ("Python syntax", check_python_sources),
        ("numerical results", check_results),
        ("English filenames", check_english_filenames),
        ("English public text", check_english_public_text),
    ]
    for label, check in checks:
        check()
        print(f"[OK] {label}")
    print("Repository verification completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
