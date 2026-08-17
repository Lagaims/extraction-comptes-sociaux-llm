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
    uv run json_to_csv.py --method marker --overwrite   (régénère au lieu d'ignorer)
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


def _covering_label_index(parent: list[str]) -> int | None:
    """Position, dans une ligne d'en-tête, du libellé qui couvre plusieurs colonnes.

    Deux signaux, dans cet ordre. Le premier est lexical — le vocabulaire du tableau
    réglementaire, seul en-tête à sous-colonnes de ce corpus. Le second ne suppose aucun
    vocabulaire : quand la ligne parente n'offre qu'un seul libellé susceptible de
    couvrir des sous-colonnes, il n'y a rien à trancher. La première colonne en est
    exclue : elle porte les raisons sociales, et y placer les sous-libellés reviendrait à
    qualifier la colonne des libellés de lignes.

    Args:
        parent: ligne d'en-tête, courte des cellules de continuation du libellé couvrant.

    Returns:
        L'indice du libellé couvrant, ou None si la ligne parente ne permet pas de
        trancher — auquel cas le repli à droite s'applique.
    """
    lexical = [j for j, c in enumerate(parent) if _GROUP_HEADER_RE.search(_norm(c))]
    if len(lexical) == 1:
        return lexical[0]
    candidates = [
        j for j, c in enumerate(parent) if j > 0 and c.strip() and not _is_numeric_cell(c)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _continuation_run_index(parent: list[str], k: int) -> int | None:
    """Position du libellé suivi d'exactement `k - 1` cellules de continuation vides.

    Cas du moteur qui a bien émis les cellules de continuation du libellé couvrant, mais
    livre quand même la sous-ligne à plat. Le trou dans la ligne parente désigne alors la
    position sans avoir à interpréter le moindre libellé.

    Args:
        parent: ligne d'en-tête, déjà à la largeur du tableau.
        k: nombre de cellules de la sous-ligne à replacer.

    Returns:
        L'indice du libellé couvrant, ou None si aucun trou de la bonne longueur n'existe
        ou si plusieurs sont candidats.
    """
    matches = [
        j
        for j in range(len(parent) - k + 1)
        if parent[j].strip() and all(not c.strip() for c in parent[j + 1 : j + k])
    ]
    return matches[0] if len(matches) == 1 else None


def _align_header_subrows(table: Table, width: int) -> Table:
    """Replace une sous-ligne d'en-tête sous le libellé qu'elle détaille.

    Deux configurations produisent une sous-ligne mal placée, selon que le moteur a émis
    ou non les cellules de continuation du libellé couvrant :

    - **ligne parente courte de `k - 1`** : les continuations manquent, un libellé couvre
      donc `k` colonnes. La ligne parente est écartée pour laisser la place, et la
      sous-ligne posée sous le libellé identifié par `_covering_label_index` ;
    - **ligne parente déjà à la largeur du tableau** : les continuations sont là, le trou
      qu'elles forment désigne la position (`_continuation_run_index`).

    Sans ce traitement, la sous-ligne est complétée à droite et atterrit en colonnes
    0..k-1 — donc sous les mauvais en-têtes, ce qui fausse l'appariement de toutes les
    colonnes du tableau. Quand la position ne peut pas être tranchée, le repli à droite
    s'applique : mieux vaut le comportement connu qu'un placement arbitraire.

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
        if len(parent) == width - (k - 1):
            j = _covering_label_index(parent)
            if j is None:
                continue
            rows[i - 1] = parent[: j + 1] + [""] * (k - 1) + parent[j + 1 :]
        elif len(parent) == width:
            j = _continuation_run_index(parent, k)
            if j is None:
                continue
        else:
            continue
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


# ── Recollage des tableaux coupés par un saut de page ─────────────────────────


def _rows_match(left: list[str], right: list[str]) -> bool:
    """Deux lignes portent-elles le même texte, à la graphie près ?"""
    return [_norm(c).strip() for c in left] == [_norm(c).strip() for c in right]


def _continuation_offset(previous: Table, candidate: Table) -> int | None:
    """Le candidat prolonge-t-il le tableau précédent, et à partir de quelle ligne ?

    Un tableau à cheval sur deux pages est livré en deux blocs par les moteurs, donc en
    deux CSV : les rangs se désynchronisent alors de ceux des annotations et toute la
    suite du SIREN est comparée de travers. Deux signaux le trahissent, à largeur égale :
    un bloc qui commence directement par des données n'a pas d'en-tête, donc n'est pas un
    tableau autonome ; un bloc qui rappelle l'en-tête du précédent le répète en tête de
    page. Un bloc portant un en-tête différent est un autre tableau.

    Args:
        previous: dernier tableau de la page précédente.
        candidate: premier tableau de la page courante.

    Returns:
        Le nombre de lignes de tête du candidat à écarter avant de le recoller (0 s'il
        reprend directement en données), ou None si les deux blocs sont bien deux
        tableaux distincts.
    """
    if not previous or not candidate:
        return None
    if _canonical_width(previous) != _canonical_width(candidate):
        return None
    # Un bloc sans ligne de données n'est pas un tableau : rien de sûr à recoller.
    if _first_data_row(previous) >= len(previous):
        return None
    header = _first_data_row(candidate)
    if header == 0:
        return 0
    if header >= len(candidate):
        return None
    if header <= len(previous) and all(
        _rows_match(previous[i], candidate[i]) for i in range(header)
    ):
        return header
    return None


def _merge_page_continuations(pages: list[list[Table]]) -> tuple[list[Table], int]:
    """Recolle le premier tableau d'une page au dernier de la précédente, s'il le prolonge.

    Le recollage est restreint aux frontières de page, seul endroit où la coupure est un
    artefact connu de la segmentation. Deux tableaux voisins d'une même page sont laissés
    tels quels : rien ne distingue alors une coupure d'une succession légitime.

    Le recollage se contente de concaténer les lignes : la mise en forme reste à
    `_normalize_grid`, qui verra le tableau entier. Rectangulariser ici serait au
    contraire nuisible, une ligne-label plus longue que les données imposant alors ses
    colonnes fantômes à la largeur canonique.

    Args:
        pages: tableaux de chaque page, dans l'ordre de lecture. Une page sans tableau
            rompt la continuité.

    Returns:
        La liste des tableaux après recollage, et le nombre de recollages effectués.
    """
    tables: list[Table] = []
    previous_page_last: int | None = None
    merges = 0
    for page in pages:
        if not page:
            previous_page_last = None
            continue
        first, *rest = page
        offset = (
            None
            if previous_page_last is None
            else _continuation_offset(tables[previous_page_last], first)
        )
        if offset is None:
            tables.append(first)
        else:
            tables[previous_page_last] = tables[previous_page_last] + first[offset:]
            merges += 1
        tables.extend(rest)
        previous_page_last = len(tables) - 1
    return tables, merges


# ── Extracteurs ───────────────────────────────────────────────────────────────


class TableExtractor(ABC):
    # Nombre de tableaux recollés lors du dernier `extract`, pour le journal du pipeline.
    merges: int = 0

    @abstractmethod
    def extract(self, data: dict) -> list[Table]:
        """Extraire les tableaux d'un JSON ; retourne une liste de matrices de chaînes."""


class MarkerTableExtractor(TableExtractor):
    """
    Extrait les tableaux HTML imbriqués produits par l'API Marker.
    Parcourt récursivement les blocs de type Table / TableGroup, page par page.
    """

    def extract(self, data: dict) -> list[Table]:
        pages = [
            [table for block in blocks for table in self._block_tables(block)]
            for blocks in self._table_blocks_by_page(data)
        ]
        tables, self.merges = _merge_page_continuations(pages)
        return tables

    @staticmethod
    def _block_tables(block: dict) -> list[Table]:
        """Grilles portées par un bloc.

        Args:
            block: bloc `Table` ou `TableGroup`.

        Returns:
            Une grille par `<table>` du fragment HTML. Un fragment sans balise `<table>`
            est enveloppé dans une ligne artificielle : s'il ne porte aucune cellule — le
            cas des blocs vides et des `<content-ref>` — il ne produit aucune grille.
        """
        html = block.get("html", "")
        if "<table" not in html.lower():
            html = f"<table><tbody><tr>{html}</tr></tbody></table>"
        return _parse_html_tables(html)

    def _table_blocks_by_page(self, node) -> list[list[dict]]:
        """Regroupe les blocs de tableau par page, dans l'ordre de lecture.

        Le regroupement par page sert au recollage des tableaux coupés par un saut de
        page. Un `TableGroup` dont la descendance porte des blocs `Table` est écarté au
        profit de ceux-ci : son `html` ne contient en principe que des pointeurs
        `<content-ref>`, mais rien dans le format ne le garantit, et le retenir en plus de
        ses enfants dupliquerait le tableau.

        Args:
            node: racine du JSON marker, ou tout sous-arbre.

        Returns:
            Une liste de blocs par bloc `Page` rencontré. Les blocs de tableau situés hors
            de toute page forment un groupe final.
        """
        pages: list[list[dict]] = []
        loose: list[dict] = []

        def walk(current, sink: list[dict]) -> None:
            if isinstance(current, list):
                for item in current:
                    walk(item, sink)
                return
            if not isinstance(current, dict):
                return
            block_type = current.get("block_type")
            children = current.get("children") or []
            if block_type == "Page":
                page: list[dict] = []
                pages.append(page)
                walk(children, page)
            elif block_type == "Table":
                # Les enfants d'un `Table` sont ses cellules : rien à y chercher.
                sink.append(current)
            elif block_type == "TableGroup":
                nested: list[dict] = []
                walk(children, nested)
                sink.extend(nested or [current])
            else:
                walk(children, sink)

        walk(node, loose)
        if loose:
            pages.append(loose)
        return pages


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
        pages = [
            [table for table in page.get("tables", []) if table] for page in data.get("pages", [])
        ]
        # Le recollage précède la normalisation : la largeur canonique et le placement des
        # sous-lignes d'en-tête se lisent mieux sur le tableau entier que sur un fragment
        # de fin de page, qui n'a ni en-tête ni forcément toutes ses colonnes remplies.
        merged, self.merges = _merge_page_continuations(pages)
        tables = []
        for table in merged:
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


def _stale_csv_paths(existing: list[str], siren: str, kept: int) -> list[str]:
    """Chemins des CSV d'un passage antérieur devenus surnuméraires.

    Une régénération produisant moins de tableaux qu'avant — c'est ce que fait le
    recollage des sauts de page — laisserait sinon les rangs excédentaires en place, et le
    dossier de sortie mélangerait deux générations de conversion.

    Args:
        existing: chemins présents dans le dossier de sortie.
        siren: radical du fichier source.
        kept: nombre de tableaux écrits par le passage courant.

    Returns:
        Les chemins `{siren}_{n}.csv` dont le rang dépasse `kept`. Les fichiers d'un autre
        radical ne sont jamais retournés, y compris quand un radical en préfixe un autre.
    """
    pattern = re.compile(rf"^{re.escape(siren)}_(\d+)\.csv$")
    stale = []
    for path in existing:
        match = pattern.match(Path(path).name)
        if match and int(match.group(1)) > kept:
            stale.append(path)
    return stale


def run_pipeline(method: str, fs: s3fs.S3FileSystem, overwrite: bool = False) -> None:
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

    ok = skipped = 0
    empty: list[str] = []
    failed: list[str] = []
    for file_key in input_files:
        raw_stem = Path(file_key).stem
        siren = raw_stem.removeprefix(strip_prefix) if strip_prefix else raw_stem
        if not overwrite and fs.exists(f"{cfg['output']}/{siren}_1.csv"):
            print(f"  [SKIP]  {siren}")
            skipped += 1
            continue

        try:
            data = _load(fs, file_key, ext)
            tables = extractor.extract(data)
        except Exception as e:
            print(f"  [ERR]   {siren}: {e}")
            failed.append(siren)
            continue

        if not tables:
            print(f"  [VIDE]  {siren}: aucun tableau détecté")
            empty.append(siren)
            continue

        for i, table in enumerate(tables, start=1):
            fs.pipe(f"{cfg['output']}/{siren}_{i}.csv", _to_csv_bytes(table))
        if overwrite:
            for path in _stale_csv_paths(
                fs.glob(f"{cfg['output']}/{siren}_*.csv"), siren, len(tables)
            ):
                fs.rm(path)

        merges = f", {extractor.merges} recollage(s) de saut de page" if extractor.merges else ""
        print(f"  [OK]    {siren}: {len(tables)} tableau(x){merges}")
        ok += 1

    print(
        f"\n[{method}] terminé — {ok} traité(s), {skipped} ignoré(s), "
        f"{len(empty)} sans tableau, {len(failed)} en erreur"
    )
    # Un fichier sans tableau ne produit aucun CSV : il disparaît de l'évaluation, qui ne
    # compare que les paires existantes. Le nommer est le seul moyen de le voir.
    if empty:
        print(f"  sans tableau, donc absents de l'évaluation : {', '.join(empty)}")
    if failed:
        print(f"  en erreur de lecture ou de conversion : {', '.join(failed)}")
    if skipped:
        print(f"  sortie déjà présente, non régénérée : {skipped} fichier(s) — voir --overwrite")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversion JSON → CSV (Marker / OpenDataLoader)")
    parser.add_argument(
        "--method",
        choices=["marker", "opendataloader", "chandra", "marker_last_work", "all"],
        default="all",
        help="Source à convertir (défaut : all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Réécrire les sorties existantes et supprimer les rangs surnuméraires. "
            "Nécessaire dès que le code de conversion change, sans quoi les fichiers déjà "
            "convertis sont ignorés et le dossier mélange deux générations."
        ),
    )
    args = parser.parse_args()

    fs = get_s3_fs()
    methods = list(SOURCES) if args.method == "all" else [args.method]
    for method in methods:
        run_pipeline(method, fs, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
