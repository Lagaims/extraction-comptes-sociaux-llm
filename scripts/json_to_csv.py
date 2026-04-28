#!/usr/bin/env python3
"""
Conversion des JSONs de sortie Marker / OpenDataLoader en tableaux CSV, vers S3.

Marker      : s3://projet-extraction-tableaux/reprise/output_marker/
               → s3://projet-extraction-tableaux/reprise/output_csv/marker/
OpenDataLoader : s3://projet-extraction-tableaux/reprise/output_opendataloader/
               → s3://projet-extraction-tableaux/reprise/output_csv/opendataloader/

Usage :
    uv run json_to_csv.py --method marker
    uv run json_to_csv.py --method opendataloader
    uv run json_to_csv.py --method all          (défaut)
"""

import argparse
import csv
import io
import json
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path

import s3fs

from extraction_common.s3 import get_s3_fs

BUCKET = "projet-extraction-tableaux"

SOURCES: dict[str, dict[str, str]] = {
    "marker": {
        "input": f"{BUCKET}/reprise/output_marker",
        "output": f"{BUCKET}/reprise/output_csv/marker",
        "ext": ".json",
    },
    "opendataloader": {
        "input": f"{BUCKET}/reprise/output_opendataloader",
        "output": f"{BUCKET}/reprise/output_csv/opendataloader",
        "ext": ".html",
    },
    "chandra": {
        "input": f"{BUCKET}/reprise/output_chandra",
        "output": f"{BUCKET}/reprise/output_csv/chandra",
        "ext": ".json",
    },
}

Table = list[list[str]]


# ── Extracteurs ───────────────────────────────────────────────────────────────

class TableExtractor(ABC):
    @abstractmethod
    def extract(self, data: dict) -> list[Table]:
        """Extraire les tableaux d'un JSON ; retourne une liste de matrices de chaînes."""


class MarkerTableExtractor(TableExtractor):
    """
    Extrait les tableaux HTML imbriqués produits par l'API Marker.
    Parcourt récursivement les blocs de type Table / TableGroup.
    """

    def extract(self, data: dict) -> list[Table]:
        tables: list[Table] = []
        for block in self._find_table_blocks(data):
            html = block.get("html", "")
            if "<table" not in html.lower():
                html = f"<table><tbody><tr>{html}</tr></tbody></table>"
            tables.extend(_parse_html_tables(html))
        return tables

    def _find_table_blocks(self, node) -> list[dict]:
        results: list[dict] = []
        if isinstance(node, list):
            for item in node:
                results.extend(self._find_table_blocks(item))
        elif isinstance(node, dict):
            if node.get("block_type") in ("Table", "TableGroup"):
                results.append(node)
            for child in node.get("children") or []:
                results.extend(self._find_table_blocks(child))
        return results


class ChandraTableExtractor(TableExtractor):
    """
    Extrait les tableaux depuis la sortie JSON de l'API Chandra.

    Format attendu :
    {
      "pages": [
        {"page": 1, "tables": [[["col1", "col2"], ["val1", "val2"]], ...]},
        ...
      ]
    }
    """

    def extract(self, data: dict) -> list[Table]:
        tables = []
        for page in data.get("pages", []):
            for table in page.get("tables", []):
                if table:
                    tables.append(table)
        return tables


class OpenDataLoaderTableExtractor(TableExtractor):
    """
    Extrait les tableaux depuis la sortie HTML d'OpenDataLoader en mode hybrid docling-fast.
    Docling reconstruit les cellules individuellement → HTML avec vraies balises <table>/<td>.
    Réutilise le même parseur HTML que Marker.
    """

    def extract(self, data: str) -> list[Table]:
        return _parse_html_tables(data)


# ── Parser HTML interne (Marker) ──────────────────────────────────────────────

class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[Table] = []
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: str = ""
        self._in_cell: bool = False

    def handle_starttag(self, tag, attrs):
        if tag in ("th", "td"):
            self._in_cell = True
            self._cell = ""
        elif tag == "br" and self._in_cell:
            self._cell += " "
        elif tag == "tr":
            self._row = []
        elif tag == "table":
            self._rows = []

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            self._in_cell = False
            self._row.append(self._cell.strip())
        elif tag == "tr" and self._row:
            self._rows.append(self._row)
        elif tag == "table" and self._rows:
            self.tables.append(self._rows)

    def handle_data(self, data):
        if self._in_cell:
            self._cell += data.replace("\n", " ")


def _parse_html_tables(html: str) -> list[Table]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.tables


# ── Sérialisation ─────────────────────────────────────────────────────────────

def _to_csv_bytes(table: Table) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, delimiter=";").writerows(table)
    return buf.getvalue().encode("utf-8-sig")


# ── Pipeline S3 ───────────────────────────────────────────────────────────────

def _load(fs: s3fs.S3FileSystem, path: str, ext: str):
    """Charge un fichier S3 : renvoie un dict (JSON) ou une str (HTML)."""
    if ext == ".json":
        with fs.open(path, "rb") as f:
            return json.load(f)
    else:
        with fs.open(path, "r", encoding="utf-8") as f:
            return f.read()


def run_pipeline(method: str, fs: s3fs.S3FileSystem) -> None:
    cfg = SOURCES[method]
    ext = cfg["ext"]
    extractors: dict[str, TableExtractor] = {
        "marker":         MarkerTableExtractor(),
        "opendataloader": OpenDataLoaderTableExtractor(),
        "chandra":        ChandraTableExtractor(),
    }
    extractor = extractors[method]

    input_files = sorted(fs.glob(f"{cfg['input']}/*{ext}"))
    print(f"[{method}] {len(input_files)} fichier(s) trouvé(s)\n")

    ok = skipped = errors = 0
    for file_key in input_files:
        siren = Path(file_key).stem
        if fs.exists(f"{cfg['output']}/{siren}_1.csv"):
            print(f"  [SKIP]  {siren}")
            skipped += 1
            continue

        try:
            data = _load(fs, file_key, ext)
            tables = extractor.extract(data)
        except Exception as e:
            print(f"  [ERR]   {siren}: {e}")
            errors += 1
            continue

        if not tables:
            print(f"  [VIDE]  {siren}: aucun tableau détecté")
            errors += 1
            continue

        for i, table in enumerate(tables, start=1):
            fs.pipe(f"{cfg['output']}/{siren}_{i}.csv", _to_csv_bytes(table))

        print(f"  [OK]    {siren}: {len(tables)} tableau(x)")
        ok += 1

    print(f"\n[{method}] terminé — {ok} traité(s), {skipped} ignoré(s), {errors} sans tableau\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Conversion JSON → CSV (Marker / OpenDataLoader)")
    parser.add_argument(
        "--method",
        choices=["marker", "opendataloader", "chandra", "all"],
        default="all",
        help="Source à convertir (défaut : all)",
    )
    args = parser.parse_args()

    fs = get_s3_fs()
    methods = list(SOURCES) if args.method == "all" else [args.method]
    for method in methods:
        run_pipeline(method, fs)


if __name__ == "__main__":
    main()
