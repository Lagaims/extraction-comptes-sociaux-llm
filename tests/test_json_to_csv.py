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
    _continuation_offset,
    _merge_page_continuations,
    _normalize_chandra_table,
    _normalize_grid,
    _parse_html_tables,
    _rectangularize,
    _stale_csv_paths,
    merge_continuations,
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
    """Deux tableaux d'en-têtes différents restent deux tableaux, page à page."""
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


# ── sous-lignes d'en-tête : positions déduites sans vocabulaire ───────────────


def test_sous_ligne_entete_placee_dans_le_trou_des_cellules_de_continuation():
    """La ligne parente est complète : le trou qu'elle porte désigne la position.

    Le moteur a émis les cellules de continuation du libellé couvrant mais livre quand
    même la sous-ligne à plat. Aucun libellé n'a besoin d'être interprété.
    """
    table = [
        ["SOCIETES", "Valeur d'inventaire", "", "Dividendes"],
        ["Brute", "Nette"],
        ["Entité A", "1 734 110", "1 700 000", "12"],
    ]
    assert _normalize_chandra_table(table)[:2] == [
        ["SOCIETES", "Valeur d'inventaire", "", "Dividendes"],
        ["", "Brute", "Nette", ""],
    ]


def test_sous_ligne_entete_trou_ambigu_reste_a_droite():
    """Deux trous de la bonne longueur : rien ne dit lequel la sous-ligne détaille."""
    table = [
        ["Filiales", "Brut", "", "Net", ""],
        ["a", "b"],
        ["Entité A", "1", "2", "3", "4"],
    ]
    assert _normalize_chandra_table(table)[1] == ["a", "b", "", "", ""]


def test_sous_ligne_entete_placee_sans_vocabulaire_si_un_seul_candidat():
    """Hors du tableau réglementaire, un unique libellé candidat suffit à trancher.

    La première colonne est exclue : elle porte les libellés de lignes, jamais un
    en-tête à sous-colonnes.
    """
    table = [
        ["SOCIETES", "Répartition"],
        ["N", "N-1"],
        ["Entité A", "1", "2"],
    ]
    assert _normalize_chandra_table(table) == [
        ["SOCIETES", "Répartition", ""],
        ["", "N", "N-1"],
        ["Entité A", "1", "2"],
    ]


def test_sous_ligne_entete_deux_candidats_sans_vocabulaire_reste_a_droite():
    table = [
        ["SOCIETES", "Alpha", "Beta"],
        ["a", "b"],
        ["Entité A", "1", "2", "3"],
    ]
    assert _normalize_chandra_table(table)[1] == ["a", "b", "", ""]


# ── recollage des tableaux coupés par un saut de page ─────────────────────────


ENTETE = ["SOCIETES", "Capital", "Résultat"]


def test_recollage_dun_bloc_qui_reprend_directement_en_donnees():
    """Un bloc sans en-tête n'est pas un tableau autonome : c'est une fin de tableau.

    Sans recollage, il part dans un CSV distinct et décale tous les rangs suivants du
    SIREN par rapport aux annotations.
    """
    data = {
        "pages": [
            {"page": 1, "tables": [[ENTETE, ["Entité A", "1 000", "250"]]]},
            {"page": 2, "tables": [[["Entité B", "2 000", "500"]]]},
        ]
    }
    extractor = ChandraTableExtractor()
    assert extractor.extract(data) == [
        [ENTETE, ["Entité A", "1 000", "250"], ["Entité B", "2 000", "500"]]
    ]
    assert extractor.merges == 1


def test_recollage_ecarte_len_tete_repete_en_tete_de_page():
    data = {
        "pages": [
            {"page": 1, "tables": [[ENTETE, ["Entité A", "1 000", "250"]]]},
            {"page": 2, "tables": [[ENTETE, ["Entité B", "2 000", "500"]]]},
        ]
    }
    assert ChandraTableExtractor().extract(data) == [
        [ENTETE, ["Entité A", "1 000", "250"], ["Entité B", "2 000", "500"]]
    ]


def test_pas_de_recollage_quand_len_tete_differe():
    """Un en-tête propre signe un autre tableau, même à largeur égale."""
    autre = ["FILIALES", "Prêts", "Cautions"]
    data = {
        "pages": [
            {"page": 1, "tables": [[ENTETE, ["Entité A", "1 000", "250"]]]},
            {"page": 2, "tables": [[autre, ["Entité B", "2 000", "500"]]]},
        ]
    }
    extractor = ChandraTableExtractor()
    assert len(extractor.extract(data)) == 2
    assert extractor.merges == 0


def test_pas_de_recollage_a_largeur_differente():
    data = {
        "pages": [
            {"page": 1, "tables": [[ENTETE, ["Entité A", "1 000", "250"]]]},
            {"page": 2, "tables": [[["Entité B", "2 000"]]]},
        ]
    }
    assert len(ChandraTableExtractor().extract(data)) == 2


def test_recollage_ne_concerne_que_le_dernier_tableau_de_la_page():
    """Le premier tableau d'une page prolonge le dernier de la précédente, pas un autre."""
    premier = [["ACTIF", "Brut", "Net"], ["Immobilisations", "10", "8"]]
    data = {
        "pages": [
            {"page": 1, "tables": [premier, [ENTETE, ["Entité A", "1 000", "250"]]]},
            {"page": 2, "tables": [[["Entité B", "2 000", "500"]]]},
        ]
    }
    tables = ChandraTableExtractor().extract(data)
    assert tables[0] == premier
    assert tables[1] == [ENTETE, ["Entité A", "1 000", "250"], ["Entité B", "2 000", "500"]]


def test_une_page_sans_tableau_rompt_la_continuite():
    data = {
        "pages": [
            {"page": 1, "tables": [[ENTETE, ["Entité A", "1 000", "250"]]]},
            {"page": 2, "tables": []},
            {"page": 3, "tables": [[["Entité B", "2 000", "500"]]]},
        ]
    }
    assert len(ChandraTableExtractor().extract(data)) == 2


def test_deux_tableaux_voisins_dune_meme_page_ne_sont_pas_recolles():
    """Sur une même page, rien ne distingue une coupure d'une succession légitime."""
    data = {
        "pages": [
            {
                "page": 1,
                "tables": [
                    [ENTETE, ["Entité A", "1 000", "250"]],
                    [["Entité B", "2 000", "500"]],
                ],
            }
        ]
    }
    assert len(ChandraTableExtractor().extract(data)) == 2


def test_recollage_marker_entre_deux_blocs_page():
    """Le recollage vaut pour marker aussi : ses tableaux sont groupés par bloc `Page`."""
    page_1 = (
        "<table><tbody><tr><th>SOCIETES</th><th>Capital</th><th>Résultat</th></tr>"
        "<tr><td>Entité A</td><td>1 000</td><td>250</td></tr></tbody></table>"
    )
    page_2 = "<table><tbody><tr><td>Entité B</td><td>2 000</td><td>500</td></tr></tbody></table>"
    data = {
        "block_type": "Document",
        "children": [
            {"block_type": "Page", "children": [{"block_type": "Table", "html": page_1}]},
            {"block_type": "Page", "children": [{"block_type": "Table", "html": page_2}]},
        ],
    }
    assert MarkerTableExtractor().extract(data) == [
        [ENTETE, ["Entité A", "1 000", "250"], ["Entité B", "2 000", "500"]]
    ]


def test_recollage_laisse_la_normalisation_trancher_la_largeur():
    """Le recollage ne rectangularise pas : la ligne-label ne doit pas fixer la largeur.

    Une grille rectangularisée avant normalisation ferait passer l'intertitre pour une
    ligne à la largeur du tableau, et ses colonnes vides s'imposeraient à toutes les
    autres lignes.
    """
    data = {
        "pages": [
            {
                "page": 1,
                "tables": [
                    [["", "", "", "A - FILIALES DÉTENUES À 50 %"], ENTETE, ["Entité A", "1", "2"]]
                ],
            },
            {"page": 2, "tables": [[["Entité B", "3", "4"]]]},
        ]
    }
    assert ChandraTableExtractor().extract(data) == [
        [
            ["A - FILIALES DÉTENUES À 50 %", "", ""],
            ENTETE,
            ["Entité A", "1", "2"],
            ["Entité B", "3", "4"],
        ]
    ]


def test_continuation_offset_refuse_un_bloc_sans_donnees():
    """Un bloc qui n'a aucune ligne de données n'est pas un tableau à recoller."""
    assert _continuation_offset([ENTETE], [["Entité B", "2 000", "500"]]) is None
    assert _continuation_offset([ENTETE, ["Entité A", "1", "2"]], [["titre"]]) is None
    assert _continuation_offset([], [["Entité B", "2 000", "500"]]) is None


def test_merge_page_continuations_compte_les_recollages():
    suite = [["Entité B", "2 000", "500"]]
    pages = [[[ENTETE, ["Entité A", "1 000", "250"]]], [suite], [suite]]
    tables, merges = _merge_page_continuations(pages)
    assert len(tables) == 1
    assert len(tables[0]) == 4
    assert merges == 2


def test_merge_continuations_recolle_jusqua_la_cible():
    """L'évaluation recolle les annotations jusqu'au compte de la prédiction.

    Sans quoi l'annotation surnuméraire n'est appariée à rien et ses cellules sortent
    du dénominateur : le score est alors calculé sur un corpus amputé, sans le dire.
    """
    debut = [ENTETE, ["Entité A", "1 000", "250"]]
    suite = [["Entité B", "2 000", "500"]]
    assert merge_continuations([debut, suite], 1) == [debut + suite]


def test_merge_continuations_sarrete_a_la_cible():
    """Deux recollages sont possibles, un seul est nécessaire : le second est laissé."""
    debut = [ENTETE, ["Entité A", "1 000", "250"]]
    suite = [["Entité B", "2 000", "500"]]
    assert merge_continuations([debut, suite, suite], 2) == [debut + suite, suite]


def test_merge_continuations_ne_fusionne_pas_deux_tableaux_distincts():
    """La cible ne prime pas sur la règle : un tableau à en-tête propre reste entier."""
    autre = [["Dénomination", "% Intérêt"], ["Entité B", "50"]]
    tables = [[ENTETE, ["Entité A", "1 000", "250"]], autre]
    assert merge_continuations(tables, 1) == tables


def test_merge_continuations_sans_cible_a_atteindre():
    """Prédiction et annotation déjà au même compte : rien n'est touché."""
    debut = [ENTETE, ["Entité A", "1 000", "250"]]
    suite = [["Entité B", "2 000", "500"]]
    assert merge_continuations([debut, suite], 2) == [debut, suite]


# ── chandra : les deux formats de sortie de l'API ─────────────────────────────


def test_chandra_html_conserve_les_fusions():
    """Le format HTML porte les fusions : elles sont développées, pas devinées.

    Le format historique, aplati par l'API, obligeait à replacer la sous-ligne au moyen
    d'heuristiques ; ici `colspan` et `rowspan` donnent la réponse.
    """
    data = {
        "metadata": {"model": "datalab-to/chandra-ocr-2", "dpi": 200},
        "pages": [
            {
                "page": 1,
                "html": (
                    "<table><tbody>"
                    "<tr><th rowspan='2'>SOCIETES</th><th rowspan='2'>Capital</th>"
                    "<th colspan='2'>Valeur d'inventaire</th></tr>"
                    "<tr><th>Brute</th><th>Nette</th></tr>"
                    "<tr><td>Entité A</td><td>877 668</td><td>1 734 110</td><td>1 700 000</td></tr>"
                    "</tbody></table>"
                ),
            }
        ],
    }
    assert ChandraTableExtractor().extract(data) == [
        [
            ["SOCIETES", "Capital", "Valeur d'inventaire", ""],
            ["", "", "Brute", "Nette"],
            ["Entité A", "877 668", "1 734 110", "1 700 000"],
        ]
    ]


def test_chandra_html_separe_les_lignes_dune_meme_cellule():
    """`<br>` devient une espace au lieu de souder les mots.

    C'est ce que l'aplatissement en amont détruisait : « Prêts etavancesconsentis ».
    """
    html = (
        "<table><tbody>"
        "<tr><th>SOCIETES</th><th>Prêts et<br/>avances<br/>consentis</th><th>Capital</th></tr>"
        "<tr><td>Entité A</td><td>1 000</td><td>250</td></tr>"
        "</tbody></table>"
    )
    grille = ChandraTableExtractor().extract({"pages": [{"page": 1, "html": html}]})
    assert grille[0][0] == ["SOCIETES", "Prêts et avances consentis", "Capital"]


def test_chandra_html_recolle_aussi_les_sauts_de_page():
    """Le recollage ne dépend pas du format d'entrée."""
    entete = "<tr><th>SOCIETES</th><th>Capital</th><th>Résultat</th></tr>"
    data = {
        "pages": [
            {
                "page": 1,
                "html": f"<table><tbody>{entete}"
                "<tr><td>Entité A</td><td>1 000</td><td>250</td></tr></tbody></table>",
            },
            {
                "page": 2,
                "html": "<table><tbody>"
                "<tr><td>Entité B</td><td>2 000</td><td>500</td></tr></tbody></table>",
            },
        ]
    }
    extractor = ChandraTableExtractor()
    assert extractor.extract(data) == [
        [ENTETE, ["Entité A", "1 000", "250"], ["Entité B", "2 000", "500"]]
    ]
    assert extractor.merges == 1


def test_chandra_les_deux_formats_donnent_la_meme_grille():
    """Sur un tableau sans fusion, l'ancien et le nouveau format se rejoignent."""
    plat = {"pages": [{"page": 1, "tables": [[ENTETE, ["Entité A", "1 000", "250"]]]}]}
    html = {
        "pages": [
            {
                "page": 1,
                "html": (
                    "<table><tbody>"
                    "<tr><th>SOCIETES</th><th>Capital</th><th>Résultat</th></tr>"
                    "<tr><td>Entité A</td><td>1 000</td><td>250</td></tr>"
                    "</tbody></table>"
                ),
            }
        ]
    }
    extractor = ChandraTableExtractor()
    assert extractor.extract(html) == extractor.extract(plat)


def test_normalize_grid_est_idempotent():
    """Une grille déjà normalisée traverse la normalisation sans changer.

    C'est ce qui permet aux grilles issues du parseur HTML — déjà normalisées — de suivre
    ensuite le même chemin que les matrices à plat.
    """
    brut = [
        ["SOCIETES", "Valeur d'inventaire", "Dividendes"],
        ["Brute", "Nette"],
        ["Entité A", "1", "2", "3"],
        ["", "", "", "intertitre"],
    ]
    une_fois = _normalize_grid(brut)
    assert _normalize_grid(une_fois) == une_fois


# ── marker : un seul tableau par `TableGroup`, garanti par le code ────────────


def test_table_group_ecarte_au_profit_de_ses_blocs_table():
    """Un `TableGroup` et ses enfants `Table` décrivent le même tableau.

    Le `html` d'un `TableGroup` ne porte en principe que des pointeurs `<content-ref>`,
    qui ne produisent aucune cellule — mais rien dans le format ne le garantit. La
    déduplication doit venir du code, pas de la forme de la sortie du moteur.
    """
    data = {
        "block_type": "Page",
        "children": [
            {
                "block_type": "TableGroup",
                "html": "<content-ref src='/page/0/Table/12'></content-ref>",
                "children": [
                    {
                        "block_type": "Table",
                        "html": "<table><tbody><tr><td>Entité A</td><td>1</td></tr></tbody></table>",
                        "children": [],
                    }
                ],
            }
        ],
    }
    assert MarkerTableExtractor().extract(data) == [[["Entité A", "1"]]]


def test_table_group_portant_lui_meme_un_table_ne_duplique_pas():
    """Même quand le groupe porte un `<table>` complet, le tableau ne sort qu'une fois."""
    html = "<table><tbody><tr><td>Entité A</td><td>1</td></tr></tbody></table>"
    data = {
        "block_type": "Page",
        "children": [
            {
                "block_type": "TableGroup",
                "html": html,
                "children": [{"block_type": "Table", "html": html, "children": []}],
            }
        ],
    }
    assert MarkerTableExtractor().extract(data) == [[["Entité A", "1"]]]


def test_table_group_sans_bloc_table_reste_retenu():
    """Un groupe qui porte seul le HTML du tableau doit être conservé."""
    data = {
        "block_type": "Page",
        "children": [
            {
                "block_type": "TableGroup",
                "html": "<table><tbody><tr><td>Entité A</td><td>1</td></tr></tbody></table>",
                "children": [],
            }
        ],
    }
    assert MarkerTableExtractor().extract(data) == [[["Entité A", "1"]]]


# ── régénération : purge des rangs surnuméraires ──────────────────────────────


def test_stale_csv_paths_ne_retient_que_les_rangs_au_dela_du_compte():
    existing = [
        "bucket/out/123456789_1.csv",
        "bucket/out/123456789_2.csv",
        "bucket/out/123456789_3.csv",
    ]
    assert _stale_csv_paths(existing, "123456789", 2) == ["bucket/out/123456789_3.csv"]
    assert _stale_csv_paths(existing, "123456789", 3) == []


def test_stale_csv_paths_ignore_les_autres_radicaux():
    """Un radical qui en préfixe un autre ne doit pas emporter ses fichiers."""
    existing = [
        "bucket/out/TAB_123_1.csv",
        "bucket/out/TAB_123_bis_4.csv",
        "bucket/out/TAB_1234_9.csv",
    ]
    assert _stale_csv_paths(existing, "TAB_123", 1) == []
