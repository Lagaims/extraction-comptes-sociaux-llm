#!/usr/bin/env python3
"""Recalcule les chiffres publiés par la page « Chiffres et cas » du site.

Les pages du site sont rédigées à la main : leurs tableaux et leurs graphiques portent des
valeurs en dur. Ce script les recalcule toutes depuis les données courantes, pour qu'une
régénération du pipeline ne laisse pas la page décrire un état révolu.

Source des cellules : `website/data/comparaisons.json`, produit par `website/build_data.py`.
Il porte les grilles annotées **et** prédites, l'appariement des lignes et des colonnes, et
le statut de chaque cellule — donc tout ce qu'il faut pour reclasser sans relire S3. Deux
mesures y échappent et lisent les CSV : la présence d'une valeur dans un autre CSV du même
SIREN, et les métriques du parquet.

La classification reprend exactement celle de `cell_status` (`website/build_data.py`), en la
poussant d'un cran : `non-appariee` se scinde selon ce qui manque — la ligne, la colonne ou
les deux — et `deplacee` selon où la valeur a atterri. C'est cette granularité que publie la
section « Décomposition des cellules attendues ».

Usage :
    uv run chiffres_site.py
    uv run chiffres_site.py --sans-s3    # saute l'entonnoir, qui relit tous les CSV
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from extraction_common.s3 import get_s3_fs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "website"))

import evaluation_extraction as E  # noqa: E402
from build_data import _digits, _norm_num  # noqa: E402

COMPARAISONS = Path(__file__).resolve().parents[1] / "website" / "data" / "comparaisons.json"
CSV_PREFIXES = {
    "marker": f"{E.BUCKET}/reprise/output_csv/marker",
    "chandra": f"{E.BUCKET}/reprise/output_csv/chandra",
}

# Ordre de publication de la décomposition, et rattachement de chaque poste à sa nature.
# « Structure » désigne un échec de placement, « Lecture » un échec de transcription : c'est
# l'opposition que la page met en avant, et elle ne se lit pas dans les statuts bruts.
DECOMPOSITION = [
    ("Récupérée", "—"),
    ("Ligne *et* colonne non appariées", "Structure"),
    ("Colonne non appariée", "Structure"),
    ("Ligne non appariée", "Structure"),
    ("Bonne ligne, colonne décalée", "Structure"),
    ("Cellule vide, valeur présente ailleurs", "Structure"),
    ("Autre décalage", "Structure"),
    ("Cellule vide, valeur nulle part", "Lecture"),
    ("Format seul", "Normalisation"),
    ("Erreur de lecture de chiffres", "Lecture"),
    ("Texte à la place du nombre", "Lecture"),
    ("Valeur franchement différente", "Lecture"),
]


def _cellule_predite(pred: list[list[str]], rmatch: dict, cmatch: dict, ligne: int, col: int):
    """Valeur prédite à la position appariée d'une cellule annotée.

    Args:
        pred: grille prédite.
        rmatch: appariement des lignes, annotation → prédiction.
        cmatch: appariement des colonnes.
        ligne: indice de ligne dans l'annotation.
        col: indice de colonne dans l'annotation.

    Returns:
        Le couple (valeur prédite, position appariée), ou (None, None) si la ligne ou la
        colonne n'est pas appariée, ou si la position tombe hors de la grille.
    """
    pr, pc = rmatch.get(str(ligne)), cmatch.get(str(col))
    if pr is None or pc is None:
        return None, None
    if pr >= len(pred) or pc >= len(pred[pr]):
        return None, None
    return pred[pr][pc], (pr, pc)


def _poste(statut: str, attendu: str, obtenu, position, pred, rmatch, cmatch, ligne, col) -> str:
    """Poste de la décomposition auquel une cellule appartient.

    Args:
        statut: statut rendu par `cell_status`.
        attendu: valeur annotée.
        obtenu: valeur prédite à la position appariée, ou None.
        position: position appariée, ou None.
        pred: grille prédite.
        rmatch, cmatch: appariements.
        ligne, col: position dans l'annotation.

    Returns:
        Le libellé du poste, tel que la page le publie.
    """
    if statut == "ok":
        return "Récupérée"
    if statut == "non-appariee":
        ligne_ok = str(ligne) in rmatch
        col_ok = str(col) in cmatch
        if not ligne_ok and not col_ok:
            return "Ligne *et* colonne non appariées"
        return "Colonne non appariée" if ligne_ok else "Ligne non appariée"
    if statut == "manquante":
        return "Cellule vide, valeur nulle part"
    if statut == "format":
        return "Format seul"
    if statut == "ocr":
        return "Erreur de lecture de chiffres"
    if statut == "differente":
        return (
            "Texte à la place du nombre" if not _digits(obtenu) else "Valeur franchement différente"
        )
    if statut == "deplacee":
        if obtenu is not None and not obtenu.strip():
            return "Cellule vide, valeur présente ailleurs"
        # La valeur est-elle sur la même ligne prédite, à une autre colonne ?
        cible = _norm_num(attendu)
        if position is not None:
            pr = position[0]
            if any(_norm_num(v) == cible for v in pred[pr]):
                return "Bonne ligne, colonne décalée"
        return "Autre décalage"
    return "Autre décalage"


def _nature_erreur(attendu: str, obtenu: str) -> str:
    """Nature d'une erreur de lecture de chiffres.

    Args:
        attendu: valeur annotée.
        obtenu: valeur prédite.

    Returns:
        Un des quatre libellés publiés par la section « Nature des erreurs ».
    """
    a, b = _digits(attendu), _digits(obtenu)
    if len(a) != len(b):
        court, long_ = sorted((a, b), key=len)
        if long_.startswith(court):
            return "Troncature en fin de cellule"
        return "Autre écart de un ou deux chiffres"
    if sorted(a) == sorted(b) and a != b:
        return "Chiffres permutés"
    if sum(1 for x, y in zip(a, b, strict=True) if x != y) == 1:
        return "Un chiffre substitué"
    return "Autre écart de un ou deux chiffres"


def _nature_ecriture(attendu: str, obtenu: str) -> str | None:
    """Nature d'un écart de convention entre deux valeurs pourtant identiques.

    Args:
        attendu: valeur annotée.
        obtenu: valeur prédite, égale après normalisation.

    Returns:
        Le libellé de l'écart, ou None si les deux chaînes sont déjà identiques.
    """
    if attendu.strip() == obtenu.strip():
        return None
    sans_espace = (re.sub(r"[\s  ]", "", attendu), re.sub(r"[\s  ]", "", obtenu))
    if sans_espace[0] == sans_espace[1]:
        return "Séparateur de milliers seul"
    if sans_espace[0].replace(",", ".") == sans_espace[1].replace(",", "."):
        return "Virgule contre point décimal"
    return "Autre convention"


def classer(data: dict) -> dict:
    """Reclasse toutes les cellules de `comparaisons.json`.

    Args:
        data: contenu de `comparaisons.json`.

    Returns:
        {methode: {"postes": Counter, "erreurs": Counter, "ecritures": Counter,
                   "total": int, "recuperees": int, "attendues_4": [(base, valeur, ok)]}}
    """
    out = {}
    for methode in data["meta"]["methods"]:
        postes: Counter = Counter()
        erreurs: Counter = Counter()
        ecritures: Counter = Counter()
        attendues_4: list[tuple[str, str, bool, bool]] = []
        for table in data["tables"]:
            resultat = table["methods"].get(methode)
            if not resultat:
                continue
            ann = resultat.get("ann") or table["ann"]
            pred = resultat["pred"]
            rmatch, cmatch = resultat["rowMatch"], resultat["colMatch"]
            valeurs_pred = {_norm_num(v) for ligne in pred for v in ligne}
            for position, statut in resultat["status"].items():
                if statut == "vide-attendue":
                    continue
                ligne, col = (int(x) for x in position.split(","))
                attendu = ann[ligne][col]
                obtenu, pos = _cellule_predite(pred, rmatch, cmatch, ligne, col)
                postes[_poste(statut, attendu, obtenu, pos, pred, rmatch, cmatch, ligne, col)] += 1
                if statut == "ocr":
                    erreurs[_nature_erreur(attendu, obtenu)] += 1
                if statut == "ok":
                    nature = _nature_ecriture(attendu, obtenu)
                    if nature:
                        ecritures[nature] += 1
                if len(_digits(attendu)) >= 4:
                    attendues_4.append(
                        (
                            E._base_stem(table["id"]),
                            attendu,
                            _norm_num(attendu) in valeurs_pred,
                            statut == "ok",
                        )
                    )
        out[methode] = {
            "postes": postes,
            "erreurs": erreurs,
            "ecritures": ecritures,
            "total": sum(postes.values()),
            "recuperees": postes["Récupérée"],
            "attendues_4": attendues_4,
        }
    return out


def _union_par_siren(fs, prefix: str) -> dict[str, set[str]]:
    """Valeurs normalisées présentes dans **tous** les CSV d'un même SIREN.

    Les CSV surnuméraires — sans annotation en face — en font partie : une valeur attendue
    peut n'exister que là, et l'entonnoir doit la compter comme lue.

    Args:
        fs: système de fichiers S3.
        prefix: dossier des CSV d'un moteur.

    Returns:
        {base_stem: ensemble des valeurs normalisées}.
    """
    union: dict[str, set[str]] = defaultdict(set)
    for chemin in fs.glob(f"{prefix}/*.csv"):
        base = E._base_stem(chemin.rsplit("/", 1)[-1].removesuffix(".csv"))
        with fs.open(chemin, "r", encoding="utf-8-sig") as f:
            for ligne in csv.reader(f, delimiter=";"):
                union[base] |= {_norm_num(v) for v in ligne}
    return union


def afficher(classement: dict, unions: dict | None, ev: pd.DataFrame | None) -> None:
    """Imprime les chiffres, section par section, dans l'ordre de la page."""
    methodes = list(classement)

    print("\n" + "=" * 78)
    print("## Erreurs de transcription — nature des erreurs")
    print("=" * 78)
    natures = [
        "Un chiffre substitué",
        "Autre écart de un ou deux chiffres",
        "Troncature en fin de cellule",
        "Chiffres permutés",
    ]
    print(f"| Nature | {' | '.join(methodes)} |")
    for nature in natures:
        print(
            f"| {nature} | {' | '.join(str(classement[m]['erreurs'][nature]) for m in methodes)} |"
        )
    print(
        f"| **Total** | {' | '.join(str(sum(classement[m]['erreurs'].values())) for m in methodes)} |"
    )

    print("\n" + "=" * 78)
    print("## Décomposition des cellules attendues")
    print("=" * 78)
    for m in methodes:
        print(f"  {m}: {classement[m]['total']} cellules non vides")
    print(f"\n| Devenir de la cellule | {' | '.join(f'{m} | %' for m in methodes)} | Nature |")
    for poste, nature in DECOMPOSITION:
        cases = []
        for m in methodes:
            n = classement[m]["postes"][poste]
            cases.append(f"{n} | {100 * n / classement[m]['total']:.1f}")
        print(f"| {poste} | {' | '.join(cases)} | {nature} |")
    print(f"| **Total** | {' | '.join(f'{classement[m]["total"]} | 100' for m in methodes)} | |")

    print("\n  Poids par nature :")
    for m in methodes:
        parts = Counter()
        for poste, nature in DECOMPOSITION:
            parts[nature] += classement[m]["postes"][poste]
        total = classement[m]["total"]
        detail = ", ".join(
            f"{nature} {100 * n / total:.1f} %" for nature, n in parts.items() if nature != "—"
        )
        print(f"    {m}: {detail}")

    print("\n" + "=" * 78)
    print("## Bien lu, écrit autrement")
    print("=" * 78)
    for m in methodes:
        c = classement[m]["ecritures"]
        n, rec = sum(c.values()), classement[m]["recuperees"]
        print(f"  {m}: {n} des {rec} cellules récupérées ({100 * n / rec:.1f} %)")
    print(f"\n| Écart | {' | '.join(methodes)} |")
    for nature in [
        "Séparateur de milliers seul",
        "Virgule contre point décimal",
        "Autre convention",
    ]:
        print(
            f"| {nature} | {' | '.join(str(classement[m]['ecritures'][nature]) for m in methodes)} |"
        )

    if unions is not None:
        print("\n" + "=" * 78)
        print("## Entonnoir d'attrition (valeurs de 4 chiffres et plus)")
        print("=" * 78)
        for m in methodes:
            valeurs = classement[m]["attendues_4"]
            total = len(valeurs)
            siren = sum(1 for base, v, _, _ in valeurs if _norm_num(v) in unions[m][base])
            bon_csv = sum(1 for _, _, dans_csv, _ in valeurs if dans_csv)
            cellule = sum(1 for _, _, _, ok in valeurs if ok)
            print(f"\n  {m}")
            for libelle, n in [
                ("Attendues (référence)", total),
                ("Lues quelque part dans le SIREN", siren),
                ("Dans le bon CSV", bon_csv),
                ("À la bonne cellule", cellule),
            ]:
                print(f"    {libelle:34} {100 * n / total:5.1f} %  · {n}")

    if ev is not None:
        print("\n" + "=" * 78)
        print("## Métriques publiées")
        print("=" * 78)
        for m in sorted(ev.methode.unique()):
            s = ev[ev.methode == m]
            cellule = s.n_recovered_numeric.sum() / s.n_numeric_cells.sum()
            print(
                f"  {m:18} n={len(s):3} col={s.col_recovery.mean():.3f} "
                f"row={s.row_recovery.mean():.3f} num={s.numeric_recovery.mean():.3f} "
                f"/cellule={cellule:.3f} total={s.total_extraction.mean():.3f} "
                f"médiane_num={s.numeric_recovery.median():.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sans-s3",
        action="store_true",
        help="saute l'entonnoir et les métriques, qui relisent S3",
    )
    args = parser.parse_args()

    if not COMPARAISONS.exists():
        raise SystemExit(
            f"{COMPARAISONS} est absent — lancer d'abord `uv run --project website "
            "python website/build_data.py`."
        )
    data = json.loads(COMPARAISONS.read_text(encoding="utf-8"))
    print(f"{data['meta']['nTables']} tableaux, moteurs {data['meta']['methods']}")
    classement = classer(data)

    unions = ev = None
    if not args.sans_s3:
        fs = get_s3_fs()
        unions = {m: _union_par_siren(fs, p) for m, p in CSV_PREFIXES.items()}
        with fs.open(E.S3_EVAL_OUTPUT, "rb") as f:
            ev = pd.read_parquet(f)

    afficher(classement, unions, ev)


if __name__ == "__main__":
    main()
