#!/usr/bin/env python3
"""
Conversion des JSONs de sortie Marker / OpenDataLoader en tableaux CSV, vers S3.

Marker      : s3://projet-extraction-tableaux/reprise/output_marker/
               → s3://projet-extraction-tableaux/reprise/output_csv/marker/
OpenDataLoader : s3://projet-extraction-tableaux/reprise/output_opendataloader/
               → s3://projet-extraction-tableaux/reprise/output_csv/opendataloader/
marker_last_work : s3://projet-extraction-tableaux/LLM_eval/response_json/
               → s3://projet-extraction-tableaux/LLM_eval/output_csv/marker_last_work/

Usage :
    uv run json_to_csv.py --method marker
    uv run json_to_csv.py --method opendataloader
    uv run json_to_csv.py --method marker_last_work
    uv run json_to_csv.py --method chandra
    uv run json_to_csv.py --method all          (défaut)
"""

import argparse
import csv
import io
import json
import re
import unicodedata
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path

import s3fs
from extraction_common.s3 import get_s3_fs

BUCKET = "projet-extraction-tableaux"

SOURCES: dict[str, dict] = {
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
    "marker_last_work": {
        "input": f"{BUCKET}/LLM_eval/response_json",
        "output": f"{BUCKET}/LLM_eval/output_csv/marker_last_work",
        "ext": ".json",
        "strip_prefix": "response_",
    },
}

Table = list[list[str]]


# Une cellule « numérique » commence par un chiffre ou un signe et ne contient que des
# chiffres, séparateurs et symboles de montant. Les libellés d'en-tête qui portent un
# appel de note (« Capital (3) ») ou une date (« 31-déc-21 ») en relèvent aussi, d'où la
# règle « au moins deux » pour qualifier une ligne de données.
_NUMERIC_CELL_RE = re.compile(r"^[(\-−+]?\d[\d\s  .,%()€$/–—-]*$")

# Dans le tableau réglementaire des filiales et participations, le seul en-tête à
# sous-colonnes est le bloc « valeurs comptables » / « valeur d'inventaire » des titres
# détenus, scindé en « Brute » / « Nette ». C'est lui que détaille une sous-ligne
# d'en-tête, et c'est sa cellule de continuation que les moteurs omettent.
_GROUP_HEADER_RE = re.compile(r"valeur|inventaire")


def _norm(value: str) -> str:
    """Minuscule sans accents, pour reconnaître un libellé quelle que soit sa graphie."""
    stripped = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return stripped.lower()


def _is_numeric_cell(value: str) -> bool:
    return bool(_NUMERIC_CELL_RE.match(value.strip()))


def _label_index(row: list[str]) -> int | None:
    """Position de l'unique cellule non vide d'une ligne, ou None s'il y en a 0 ou ≥2."""
    filled = [i for i, c in enumerate(row) if c.strip()]
    return filled[0] if len(filled) == 1 else None


def _first_data_row(table: Table) -> int:
    """Rang de la première ligne de données, c'est-à-dire d'au moins deux nombres.

    Args:
        table: grille brute.

    Returns:
        L'indice de la première ligne de données, ou `len(table)` s'il n'y en a aucune.
    """
    for i, row in enumerate(table):
        if sum(1 for c in row if _is_numeric_cell(c)) >= 2:
            return i
    return len(table)


def _canonical_width(table: Table) -> int:
    """Largeur du tableau, mesurée sur les seules lignes porteuses de plusieurs cellules.

    Une ligne-label — un intertitre de section, seule cellule non vide de sa ligne — ne
    doit pas fixer la largeur : certains moteurs la placent en fin de ligne, ce qui
    ajouterait autant de colonnes fantômes à tout le tableau.

    Args:
        table: grille brute.

    Returns:
        La largeur retenue, ou 0 pour une grille vide.
    """
    body = [row for row in table if _label_index(row) is None and row]
    return max((len(r) for r in body), default=max((len(r) for r in table), default=0))


def _align_header_subrows(table: Table, width: int) -> Table:
    """Replace une sous-ligne d'en-tête sous le libellé qu'elle détaille.

    Une sous-ligne de `k` cellules dont la ligne parente est courte d'exactement `k - 1`
    signale un libellé couvrant `k` colonnes dont les cellules de continuation n'ont pas
    été émises. Complétée à droite, elle atterrit en colonnes 0..k-1 — donc sous les
    mauvais en-têtes, ce qui fausse l'appariement de toutes les colonnes du tableau.

    Le libellé couvrant est identifié par `_GROUP_HEADER_RE`, et seulement s'il est
    **unique** dans la ligne parente : à défaut le repli à droite s'applique, faute de
    savoir laquelle des positions candidates est la bonne.

    Args:
        table: grille brute.
        width: largeur canonique.

    Returns:
        La grille, sous-lignes d'en-tête repositionnées.
    """
    rows = [list(r) for r in table]
    for i in range(1, _first_data_row(rows)):
        row = rows[i]
        k = len(row)
        if not (2 <= k < width) or any(_is_numeric_cell(c) for c in row):
            continue
        parent = rows[i - 1]
        if len(parent) != width - (k - 1):
            continue
        matches = [j for j, c in enumerate(parent) if _GROUP_HEADER_RE.search(_norm(c))]
        if len(matches) != 1:
            continue
        j = matches[0]
        rows[i - 1] = parent[: j + 1] + [""] * (k - 1) + parent[j + 1 :]
        rows[i] = [""] * j + row + [""] * (width - j - k)
    return rows


def _normalize_grid(table: Table) -> Table:
    """Met une grille extraite en forme rectangulaire, colonnes alignées.

    Trois règles, dans l'ordre : largeur mesurée hors lignes-labels, sous-lignes
    d'en-tête replacées sous le libellé qu'elles détaillent, lignes-labels ramenées en
    première colonne. Ce qui reste court est complété à droite, faute de mieux.

    Args:
        table: grille brute issue d'un extracteur.

    Returns:
        Grille rectangulaire.
    """
    if not table:
        return table
    width = _canonical_width(table)
    rows = _align_header_subrows(table, width)

    normalized: Table = []
    for row in rows:
        label = _label_index(row)
        # Un intertitre occupe la ligne entière : sa colonne d'origine ne porte aucune
        # information, et la conserver au-delà de la largeur du tableau ajouterait des
        # colonnes vides à toutes les autres lignes.
        if label is not None and len(row) > width:
            normalized.append([row[label]] + [""] * (width - 1))
        else:
            normalized.append(row)
    return _rectangularize(normalized)


def _rectangularize(table: Table) -> Table:
    """Complète les lignes courtes pour que toutes aient la largeur de la plus longue.

    Chaque extracteur rend une grille rectangulaire : c'est ici, et seulement ici, que
    l'on sait d'où viennent les cellules manquantes. `_load_csv` côté évaluation ne voit
    qu'un CSV et ne peut que compléter à droite — laisser la grille irrégulière revient
    donc à lui déléguer une décision de structure qu'il n'a pas les moyens de prendre.

    Args:
        table: grille éventuellement irrégulière.

    Returns:
        La même grille, toutes lignes portées à la largeur maximale par des cellules
        vides à droite. Une grille déjà rectangulaire est retournée inchangée.
    """
    if not table:
        return table
    width = max(len(row) for row in table)
    return [row + [""] * (width - len(row)) for row in table]


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


def _normalize_chandra_table(table: Table) -> Table | None:
    """Écarte un tableau Chandra sans données, met les autres en forme.

    La mise en forme est celle de `_normalize_grid`, commune à tous les moteurs. Seul le
    rejet est propre à chandra : ses pages produisent des blocs qui ne sont pas des
    tableaux, et un bloc dont aucune ligne ne porte deux cellules n'en est pas un.

    Args:
        table: tableau brut d'une page chandra.

    Returns:
        La grille rectangulaire, ou None si le tableau ne contient aucune ligne de
        données (≥ 2 cellules non vides).
    """
    data_rows = [row for row in table if sum(1 for c in row if c.strip()) > 1]
    if not data_rows:
        return None
    return _normalize_grid(table)


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
                    normalized = _normalize_chandra_table(table)
                    if normalized:
                        tables.append(normalized)
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
    """Convertit un tableau HTML en matrice de chaînes, fusions comprises.

    Les fusions étant traitées ligne par ligne, une ligne dont le HTML compte moins de
    cellules que les autres ressort plus courte : c'est `_parse_html_tables` qui achève
    la grille en la passant par `_rectangularize`.

    `colspan` est développé en cellules vides à droite ; `rowspan` est reporté sur les
    lignes suivantes via `_carried`. Sans ce report, chaque ligne suivant une cellule
    fusionnée verticalement perd une cellule et tout ce qui la suit glisse d'un cran à
    gauche — décalage qui se propage ensuite à l'ensemble du tableau. Les en-têtes des
    tableaux de filiales et participations en dépendent : 77 % des documents marker du
    corpus contiennent au moins un `rowspan`.

    Une cellule fusionnée ne porte sa valeur qu'à sa position d'origine ; les positions
    de continuation reçoivent une chaîne vide, dans les deux directions. C'est la
    convention des annotations de référence, où une fusion Excel n'écrit la valeur que
    dans sa première cellule, et celle déjà appliquée à `colspan`.

    Le choix n'est pas cosmétique : mesuré sur les 69 paires du corpus `reprise/`, il
    porte la récupération numérique de 42,0 % (sans report) à 48,4 %, tandis que
    répéter la valeur sur les lignes couvertes la fait tomber à 29,7 % — un libellé
    dupliqué rend les lignes indiscernables à l'appariement des en-têtes.
    """

    def __init__(self):
        super().__init__()
        self.tables: list[Table] = []
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: str = ""
        self._in_cell: bool = False
        self._colspan: int = 1
        self._rowspan: int = 1
        # {index de colonne: nombre de lignes que la fusion couvre encore}
        self._carried: dict[int, int] = {}

    @staticmethod
    def _span(value) -> int:
        """Lit un attribut colspan/rowspan.

        Args:
            value: valeur brute de l'attribut, éventuellement absente ou invalide.

        Returns:
            L'entier lu, ramené à 1 s'il est absent, non numérique ou < 1.
        """
        try:
            span = int(value)
        except (ValueError, TypeError):
            return 1
        return max(span, 1)

    def _fill_carried(self) -> None:
        """Occupe les positions de la ligne courante tenues par un `rowspan` en cours."""
        while len(self._row) in self._carried:
            self._row.append("")

    def _close_row(self) -> None:
        """Termine la ligne courante : positions reportées restantes, puis décompte.

        Une cellule reportée peut se situer au-delà de la dernière cellule écrite dans
        le HTML — la ligne est alors complétée par des cellules vides jusqu'à elle,
        sinon la position serait décalée sur toutes les lignes suivantes.
        """
        if self._carried:
            last = max(self._carried)
            while len(self._row) <= last:
                self._fill_carried()
                if len(self._row) <= last and len(self._row) not in self._carried:
                    self._row.append("")
        self._fill_carried()

        for col in list(self._carried):
            self._carried[col] -= 1
            if self._carried[col] <= 0:
                del self._carried[col]

    def handle_starttag(self, tag, attrs):
        if tag in ("th", "td"):
            self._in_cell = True
            self._cell = ""
            attrs_dict = dict(attrs)
            self._colspan = self._span(attrs_dict.get("colspan", 1))
            self._rowspan = self._span(attrs_dict.get("rowspan", 1))
        elif tag == "br" and self._in_cell:
            self._cell += " "
        elif tag == "tr":
            self._row = []
            # Une fusion verticale ouverte sur une ligne précédente occupe déjà le
            # début de celle-ci : ces positions sont pourvues avant la première cellule.
            self._fill_carried()
        elif tag == "table":
            self._rows = []
            self._carried = {}

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            self._in_cell = False
            value = self._cell.strip()
            start = len(self._row)
            for offset in range(self._colspan):
                self._row.append(value if offset == 0 else "")
            if self._rowspan > 1:
                # Le compteur vaut le rowspan entier, pas rowspan - 1 : `_close_row`
                # décompte aussi la ligne de déclaration, et le report doit lui survivre
                # pour couvrir les rowspan - 1 lignes suivantes.
                for offset in range(self._colspan):
                    self._carried[start + offset] = self._rowspan
            self._colspan = 1
            self._rowspan = 1
            self._fill_carried()
        elif tag == "tr":
            self._close_row()
            if self._row:
                self._rows.append(self._row)
        elif tag == "table" and self._rows:
            self.tables.append(self._rows)
            self._carried = {}

    def handle_data(self, data):
        if self._in_cell:
            self._cell += data.replace("\n", " ")


def _parse_html_tables(html: str) -> list[Table]:
    """Parse un fragment HTML et retourne ses tableaux, chacun rectangulaire.

    Args:
        html: fragment HTML pouvant contenir plusieurs `<table>`.

    Returns:
        Une grille de chaînes par `<table>` rencontrée.
    """
    parser = _TableHTMLParser()
    parser.feed(html)
    return [_normalize_grid(table) for table in parser.tables]


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
    strip_prefix = cfg.get("strip_prefix", "")
    extractors: dict[str, TableExtractor] = {
        "marker": MarkerTableExtractor(),
        "marker_last_work": MarkerTableExtractor(),
        "opendataloader": OpenDataLoaderTableExtractor(),
        "chandra": ChandraTableExtractor(),
    }
    extractor = extractors[method]

    input_files = sorted(fs.glob(f"{cfg['input']}/*{ext}"))
    print(f"[{method}] {len(input_files)} fichier(s) trouvé(s)\n")

    ok = skipped = errors = 0
    for file_key in input_files:
        raw_stem = Path(file_key).stem
        siren = raw_stem.removeprefix(strip_prefix) if strip_prefix else raw_stem
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
        choices=["marker", "opendataloader", "chandra", "marker_last_work", "all"],
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
