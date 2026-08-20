"""Tests de la logique pure d'évaluation (normalisation et détection de numériques).

Ces fonctions conditionnent directement les métriques `numeric_recovery` et
`total_extraction` : une régression ici fausse silencieusement l'évaluation.
"""

import pytest
from evaluation_extraction import (
    _is_numeric,
    _lev_similarity,
    _looks_numeric,
    _normalize_numeric_str,
    _rank,
    _unify_dashes,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("25 000", "25000"),  # espace séparateur de milliers
        ("25 000", "25000"),  # espace insécable (fréquent dans les scans)
        ("100%", "1"),
        ("66,67%", "0.6667"),
        ("1234", "1234"),
        ("Capital social", "Capital social"),
    ],
)
def test_normalize_numeric_str(value, expected):
    assert _normalize_numeric_str(value) == expected


@pytest.mark.parametrize(
    "value",
    ["1 234,5", "-42", "", "   "],
)
def test_is_numeric_accepte(value):
    assert _is_numeric(value)


def test_is_numeric_refuse_du_texte():
    assert not _is_numeric("Capital social")


@pytest.mark.parametrize(
    "value",
    ["(14)", "15,24 €", "344 369 NOK", "100,00%", "NC", "n/a", "-"],
)
def test_looks_numeric_tolere_unites_et_placeholders(value):
    assert _looks_numeric(value)


@pytest.mark.parametrize(
    "value",
    ["Capital social", "Total du bilan", "Exercice N-1 (12 mois)"],
)
def test_looks_numeric_refuse_les_entetes(value):
    assert not _looks_numeric(value)


@pytest.mark.parametrize(
    ("stem", "expected"),
    [("790256671_2", 2), ("TAB_301462602_10", 10), ("sans_rang", 0)],
)
def test_rank(stem, expected):
    """Le tri lexicographique placerait `_10` avant `_2`, l'appariement se ferait de travers."""
    assert _rank(stem) == expected


# ── équivalence des tirets ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("annote", "predit"),
    [
        ("-", "—"),  # cadratin : la marque d'absence la plus courante des moteurs
        ("-", "–"),  # demi-cadratin
        ("-", "−"),  # signe moins Unicode
        ("-30 000", "−30 000"),  # signe négatif devant un montant
        ("- 30 000", "— 30 000"),
    ],
)
def test_les_variantes_de_tiret_sont_equivalentes(annote, predit):
    """Une marque d'absence ou un signe négatif ne doit pas dépendre du glyphe.

    L'annotation écrit « - » là où les moteurs rendent « — » : 14 cellules de
    `TAB_552096281_2` étaient comptées « texte à la place du nombre » pour cette seule
    raison.
    """
    assert _normalize_numeric_str(annote) == _normalize_numeric_str(predit)
    assert _unify_dashes(annote.strip()) == _unify_dashes(predit.strip())


def test_le_tiret_demi_cadratin_ne_casse_plus_l_appariement_des_libelles():
    """« A - FILIALES » et « A – FILIALES » désignent la même colonne.

    Sans unification, la similarité tombe à 0,26 — sous le seuil de 0,5 — et la colonne
    n'est appariée à rien.
    """
    assert _lev_similarity("A - FILIALES DETENUES", "A – FILIALES DETENUES") == 1.0


def test_l_unification_ne_touche_que_les_tirets():
    """Le reste du texte est rendu tel quel, accents et casse compris."""
    assert _unify_dashes("Prêts et avances — Société") == "Prêts et avances - Société"
    assert _unify_dashes("Capital social") == "Capital social"
