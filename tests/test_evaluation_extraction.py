"""Tests de la logique pure d'évaluation (normalisation et détection de numériques).

Ces fonctions conditionnent directement les métriques `numeric_recovery` et
`total_extraction` : une régression ici fausse silencieusement l'évaluation.
"""

import pandas as pd
import pytest
from evaluation_extraction import (
    _is_numeric,
    _looks_numeric,
    _merge_split_annotations,
    _normalize_numeric_str,
    _rank,
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


# ── Recollage des annotations coupées par un saut de page ─────────────────────

ENTETE = ["SOCIETES", "Capital", "Résultat"]


def _df(rows: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, dtype=str)


def test_annotations_recollees_quand_la_prediction_en_compte_moins():
    """Un tableau annoté en deux fichiers, prédit en un seul, est recollé côté annotation.

    Sans cette symétrie l'annotation `_2` n'est appariée à rien : ses cellules quittent
    le dénominateur et le score porte sur un corpus amputé.
    """
    anns = [_df([ENTETE, ["Entité A", "1 000", "250"]]), _df([["Entité B", "2 000", "500"]])]
    merged = _merge_split_annotations(anns, 1)
    assert len(merged) == 1
    assert merged[0].values.tolist() == [
        ENTETE,
        ["Entité A", "1 000", "250"],
        ["Entité B", "2 000", "500"],
    ]


def test_annotations_intactes_quand_les_comptes_concordent():
    anns = [_df([ENTETE, ["Entité A", "1 000", "250"]]), _df([["Entité B", "2 000", "500"]])]
    assert _merge_split_annotations(anns, 2) is anns


def test_annotations_distinctes_jamais_fusionnees():
    """Sous-détection du moteur : deux vrais tableaux ne sont pas recollés pour autant."""
    anns = [
        _df([ENTETE, ["Entité A", "1 000", "250"]]),
        _df([["Dénomination", "% Intérêt"], ["Entité B", "50"]]),
    ]
    assert _merge_split_annotations(anns, 1) is anns


@pytest.mark.parametrize(
    ("stem", "expected"),
    [("790256671_2", 2), ("TAB_301462602_10", 10), ("sans_rang", 0)],
)
def test_rank(stem, expected):
    """Le tri lexicographique placerait `_10` avant `_2`, l'appariement se ferait de travers."""
    assert _rank(stem) == expected
