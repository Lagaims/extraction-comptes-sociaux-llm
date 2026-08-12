"""Tests de la logique pure d'évaluation (normalisation et détection de numériques).

Ces fonctions conditionnent directement les métriques `numeric_recovery` et
`total_extraction` : une régression ici fausse silencieusement l'évaluation.
"""

import pytest
from evaluation_extraction import _is_numeric, _looks_numeric, _normalize_numeric_str


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
