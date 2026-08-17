"""Tests du parseur HTML de `json_to_csv`, sans aucun accès S3.

Le parseur produit la grille sur laquelle repose toute la chaîne de mesure : un
décalage d'une colonne introduit ici se propage à l'appariement des en-têtes puis aux
métriques. Les cas couverts sont ceux réellement rencontrés dans les tableaux de
filiales et participations du corpus : en-têtes sur deux niveaux, libellé de société
fusionné sur plusieurs lignes, cellules fusionnées dans les deux sens.
"""

import pytest
from json_to_csv import (
    ChandraTableExtractor,
    MarkerTableExtractor,
    _normalize_chandra_table,
    _parse_html_tables,
    _rectangularize,
)


def parse_one(html: str) -> list[list[str]]:
    """Parse un HTML contenant un seul tableau et retourne sa grille."""
    tables = _parse_html_tables(html)
    assert len(tables) == 1, f"attendu 1 tableau, obtenu {len(tables)}"
    return tables[0]


# ── rowspan ───────────────────────────────────────────────────────────────────


def test_rowspan_reserve_la_position_sur_les_lignes_couvertes():
    """Une cellule `rowspan=2` occupe sa colonne sur les deux lignes.

    Sans report, la deuxième ligne ne contient que ['1 000', '250'] : le montant se
    retrouve dans la colonne du libellé et tout le reste glisse d'un cran à gauche.

    La position de continuation reçoit une chaîne vide, pas la valeur répétée : c'est la
    convention des annotations de référence (une fusion Excel n'écrit la valeur que dans
    sa première cellule) et celle déjà appliquée à `colspan`.
    """
    html = """
    <table><tbody>
      <tr><td rowspan="2">Entité A</td><td>2022</td><td>500</td></tr>
      <tr><td>1 000</td><td>250</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [
        ["Entité A", "2022", "500"],
        ["", "1 000", "250"],
    ]


def test_rowspan_au_milieu_de_la_ligne():
    """Une fusion verticale hors première colonne décale aussi ce qui la suit."""
    html = """
    <table><tbody>
      <tr><td>Entité A</td><td rowspan="2">EUR</td><td>500</td></tr>
      <tr><td>Entité B</td><td>250</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [
        ["Entité A", "EUR", "500"],
        ["Entité B", "", "250"],
    ]


def test_rowspan_expire_apres_le_nombre_de_lignes_annonce():
    """Le report cesse exactement après les lignes couvertes, sans déborder."""
    html = """
    <table><tbody>
      <tr><td rowspan="2">Groupe</td><td>a</td></tr>
      <tr><td>b</td></tr>
      <tr><td>Entité C</td><td>c</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [
        ["Groupe", "a"],
        ["", "b"],
        ["Entité C", "c"],
    ]


def test_rowspan_de_trois_lignes():
    html = """
    <table><tbody>
      <tr><td rowspan="3">Total</td><td>1</td></tr>
      <tr><td>2</td></tr>
      <tr><td>3</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [["Total", "1"], ["", "2"], ["", "3"]]


def test_rowspan_au_dela_de_la_derniere_cellule_ecrite():
    """Une position reportée peut suivre la dernière cellule du HTML de la ligne.

    La ligne doit être complétée jusqu'à cette position, sinon la colonne reportée se
    décale sur toutes les lignes suivantes.
    """
    html = """
    <table><tbody>
      <tr><td>a</td><td>b</td><td rowspan="2">note</td></tr>
      <tr><td>c</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [
        ["a", "b", "note"],
        ["c", "", ""],
    ]


def test_rowspan_et_colspan_combines():
    """Une cellule fusionnée dans les deux sens occupe un bloc rectangulaire."""
    html = """
    <table><tbody>
      <tr><td colspan="2" rowspan="2">Capitaux propres</td><td>2022</td></tr>
      <tr><td>2021</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [
        ["Capitaux propres", "", "2022"],
        ["", "", "2021"],
    ]


def test_entete_sur_deux_niveaux_reste_aligne():
    """Cas réel : « Brute | Nette » sous « Valeur d'inventaire », le reste en rowspan.

    C'est la structure des tableaux de filiales du corpus. Les sous-en-têtes doivent
    tomber sous la colonne qu'ils qualifient, et les montants sous leur sous-en-tête.
    """
    html = """
    <table><tbody>
      <tr>
        <th rowspan="2">SOCIETES</th>
        <th rowspan="2">Capital</th>
        <th colspan="2">Valeur d'inventaire</th>
      </tr>
      <tr><th>Brute</th><th>Nette</th></tr>
      <tr><td>Entité A</td><td>877 668</td><td>1 734 110</td><td>1 734 110</td></tr>
    </tbody></table>
    """
    grille = parse_one(html)
    assert grille == [
        ["SOCIETES", "Capital", "Valeur d'inventaire", ""],
        ["", "", "Brute", "Nette"],
        ["Entité A", "877 668", "1 734 110", "1 734 110"],
    ]
    # L'invariant qui compte : toutes les lignes ont la même largeur, donc les colonnes
    # de données sont alignées avec leur sous-en-tête.
    assert len({len(ligne) for ligne in grille}) == 1


# ── colspan (comportement préexistant, protégé contre les régressions) ────────


@pytest.mark.parametrize(
    ("attribut", "largeur_attendue"),
    [
        ('colspan="1"', 2),
        ('colspan="3"', 4),
        ("", 2),
        ('colspan="0"', 2),  # valeur invalide ramenée à 1
        ('colspan="abc"', 2),  # idem
    ],
)
def test_colspan_developpe_en_cellules_vides(attribut, largeur_attendue):
    html = f"<table><tbody><tr><td {attribut}>x</td><td>y</td></tr></tbody></table>"
    ligne = parse_one(html)[0]
    assert len(ligne) == largeur_attendue
    assert ligne[0] == "x"
    assert ligne[-1] == "y"


def test_rowspan_invalide_ne_reporte_rien():
    html = """
    <table><tbody>
      <tr><td rowspan="0">a</td><td>b</td></tr>
      <tr><td>c</td><td>d</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [["a", "b"], ["c", "d"]]


# ── autres comportements du parseur ──────────────────────────────────────────


def test_br_devient_une_espace_dans_la_cellule():
    html = "<table><tbody><tr><td>Entité A<br/>Paris</td></tr></tbody></table>"
    assert parse_one(html) == [["Entité A Paris"]]


def test_lignes_vides_ignorees():
    html = "<table><tbody><tr></tr><tr><td>a</td></tr></tbody></table>"
    assert parse_one(html) == [["a"]]


def test_plusieurs_tableaux_sont_independants():
    """Un `rowspan` ouvert dans un tableau ne doit pas fuir dans le suivant."""
    html = """
    <table><tbody><tr><td rowspan="5">A</td><td>1</td></tr></tbody></table>
    <table><tbody><tr><td>B</td><td>2</td></tr></tbody></table>
    """
    tables = _parse_html_tables(html)
    assert tables == [[["A", "1"]], [["B", "2"]]]


def test_marker_extractor_traverse_les_blocs_imbriques():
    data = {
        "children": [
            {
                "block_type": "TableGroup",
                "html": (
                    '<table><tbody><tr><td rowspan="2">Entité A</td><td>1</td></tr>'
                    "<tr><td>2</td></tr></tbody></table>"
                ),
                "children": [],
            }
        ]
    }
    assert MarkerTableExtractor().extract(data) == [[["Entité A", "1"], ["", "2"]]]


# ── rectangularité de la grille produite ─────────────────────────────────────


def test_ligne_courte_completee_a_droite():
    """Une ligne comptant moins de cellules que les autres est complétée sur place.

    Sans cela, la grille sort irrégulière et c'est `_load_csv`, côté évaluation, qui
    décide où vont les cellules manquantes — alors qu'il ne voit qu'un CSV et n'a aucun
    moyen de savoir d'où elles viennent.
    """
    html = """
    <table><tbody>
      <tr><td>a</td><td>b</td><td>c</td></tr>
      <tr><td>1</td></tr>
    </tbody></table>
    """
    assert parse_one(html) == [
        ["a", "b", "c"],
        ["1", "", ""],
    ]


def test_toutes_les_grilles_produites_sont_rectangulaires():
    """L'invariant de sortie du parseur, sur un tableau mêlant fusions et lignes courtes."""
    html = """
    <table><tbody>
      <tr><th rowspan="2">SOCIETES</th><th colspan="2">Valeur d'inventaire</th><th>Note</th></tr>
      <tr><th>Brute</th><th>Nette</th></tr>
      <tr><td>Entité A</td><td>1</td></tr>
    </tbody></table>
    """
    grille = parse_one(html)
    assert len({len(ligne) for ligne in grille}) == 1


def test_rectangularize_laisse_une_grille_deja_reguliere_inchangee():
    grille = [["a", "b"], ["c", "d"]]
    assert _rectangularize(grille) == grille
    assert _rectangularize([]) == []


# ── sous-lignes d'en-tête : repositionnement sous le libellé détaillé ─────────


def test_sous_ligne_entete_replacee_sous_le_libelle_couvrant():
    """Cas réel chandra : « Brute | Nette » sous « Valeur d'inventaire des titres ».

    La ligne parente compte 11 libellés pour 12 colonnes : un libellé en couvre deux et
    sa cellule de continuation n'a pas été émise. Complétée à droite, la sous-ligne
    tomberait en colonnes 0-1, sous « SOCIETES » et « Capital ».
    """
    table = [
        ["SOCIETES", "Capital", "Valeur d'inventaire", "Dividendes"],
        ["Brute", "Nette"],
        ["Entité A", "877 668", "1 734 110", "1 700 000", "12"],
    ]
    assert _normalize_chandra_table(table) == [
        ["SOCIETES", "Capital", "Valeur d'inventaire", "", "Dividendes"],
        ["", "", "Brute", "Nette", ""],
        ["Entité A", "877 668", "1 734 110", "1 700 000", "12"],
    ]


def test_sous_ligne_entete_de_trois_colonnes():
    """Le décalage vaut k-1, quel que soit le nombre de sous-colonnes."""
    table = [
        ["Filiales", "Valeurs comptables", "Quote-part"],
        ["Brute", "Nette", "Fair Value"],
        ["Entité A", "1", "2", "3", "4"],
    ]
    assert _normalize_chandra_table(table) == [
        ["Filiales", "Valeurs comptables", "", "", "Quote-part"],
        ["", "Brute", "Nette", "Fair Value", ""],
        ["Entité A", "1", "2", "3", "4"],
    ]


def test_sous_ligne_entete_libelle_couvrant_ambigu_reste_a_droite():
    """Deux libellés candidats : impossible de trancher, le repli à droite s'applique.

    Mieux vaut le comportement connu qu'un placement arbitraire : ici « Valeur brute »
    et « Valeur nette » sont déjà deux colonnes distinctes, et rien ne dit laquelle la
    sous-ligne détaillerait.
    """
    table = [
        ["Filiales", "Valeur brute", "Valeur nette", "Dividendes"],
        ["a", "b"],
        ["Entité A", "1", "2", "3", "4"],
    ]
    assert _normalize_chandra_table(table)[1] == ["a", "b", "", "", ""]


def test_sous_ligne_entete_non_appliquee_si_la_ligne_parente_est_complete():
    """Une ligne parente déjà à la bonne largeur ne signale aucun libellé couvrant."""
    table = [
        ["SOCIETES", "Capital", "Valeur d'inventaire"],
        ["Brute", "Nette"],
        ["Entité A", "877 668", "1 734 110"],
    ]
    assert _normalize_chandra_table(table) == [
        ["SOCIETES", "Capital", "Valeur d'inventaire"],
        ["Brute", "Nette", ""],
        ["Entité A", "877 668", "1 734 110"],
    ]


def test_ligne_de_donnees_courte_reste_completee_a_droite():
    """Le repositionnement ne concerne que la zone d'en-tête, jamais les données."""
    table = [
        ["SOCIETES", "Capital", "Valeur d'inventaire"],
        ["Entité A", "877 668"],
        ["Entité B", "1 000", "2 000"],
    ]
    assert _normalize_chandra_table(table)[1] == ["Entité A", "877 668", ""]


# ── chandra : rejet des blocs sans données ────────────────────────────────────


def test_chandra_ignore_les_tableaux_sans_ligne_de_donnees():
    assert _normalize_chandra_table([["titre seul"], ["autre"]]) is None


def test_chandra_ligne_label_plus_longue_que_les_donnees():
    """Une ligne-label ne fixe pas la largeur du tableau.

    Certains moteurs émettent l'intertitre de section en fin de ligne, avec autant de
    cellules vides avant lui. Retenu comme largeur, il ajouterait ces colonnes fantômes
    à toutes les lignes : c'est l'inverse qui doit se produire, l'intertitre revient en
    première colonne et le tableau garde la largeur de ses données.
    """
    table = [
        ["", "", "", "intertitre"],
        ["Entité A", "877 668"],
    ]
    assert _normalize_chandra_table(table) == [
        ["intertitre", ""],
        ["Entité A", "877 668"],
    ]


def test_chandra_extractor_parcourt_les_pages():
    data = {
        "pages": [
            {"page": 1, "tables": [[["a", "b"], ["1", "2"]]]},
            {"page": 2, "tables": [[["c", "d"], ["3", "4"]]]},
        ]
    }
    assert ChandraTableExtractor().extract(data) == [
        [["a", "b"], ["1", "2"]],
        [["c", "d"], ["3", "4"]],
    ]
