#!/usr/bin/env python3
"""Vérifie que `data/comparaisons.json` ne contient aucune identité en clair.

Contrôle la *sortie*, pas le code qui l'a produite : c'est ce qui rend le test utile.
Si `build_data.py` évolue et cesse de pseudonymiser une grille, un champ ou une nouvelle
méthode, la règle est violée dans le JSON et ce script échoue — même si le code semble
correct à la lecture.

Portée du contrôle, à connaître pour ne pas lui prêter plus qu'il ne vaut :

- il vérifie que tout libellé conservé en clair satisfait la règle de vocabulaire, que
  les identifiants de tableaux sont neutres, et qu'aucun jeton de 9 chiffres isolé ne
  traîne dans un libellé textuel ;
- il ne peut pas détecter qu'une raison sociale est composée uniquement de mots du
  vocabulaire comptable (« Société Générale de Participations » serait conservée si
  « generale » entrait un jour dans `_VOCAB`). C'est pourquoi `_VOCAB` ne doit jamais
  être étendu pour faire passer ce test — voir website/README.md.

Usage :
    uv run --project website python website/check_anonymisation.py
    uv run --project website python website/check_anonymisation.py --list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# `_NON_ENTITY_RE` est importée plutôt que redéfinie : c'est une exemption de politique
# explicite (code devise ISO, forme juridique — « XOF », « SAS » n'identifient personne),
# pas une heuristique à revalider. Les deux autres règles, elles, sont réappliquées ici
# sur la sortie, indépendamment du chemin de code qui l'a produite.
from build_data import _NON_ENTITY_RE, _STOPWORDS, _VOCAB, _norm_label

DATA_PATH = Path(__file__).parent / "data" / "comparaisons.json"

# Un libellé « textuel » comporte au moins trois lettres consécutives : les montants,
# codes devise et fragments de ponctuation sont hors périmètre.
_TEXTUAL_RE = re.compile(r"[A-Za-zÀ-ÿ]{3}")
_PSEUDONYM_RE = re.compile(r"^Entité \d{2,}(·[a-z])?$")
_TABLE_ID_RE = re.compile(r"^TAB-\d{2,}(_\d+)?$")
# Un SIREN est une suite de 9 chiffres non collée à d'autres chiffres. Les montants
# atteignent cette longueur, d'où la restriction aux libellés textuels.
_SIREN_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")


def collect_cells(payload: dict) -> set[str]:
    """Rassemble toutes les cellules de toutes les grilles du jeu de données.

    Args:
        payload: contenu de comparaisons.json.

    Returns:
        Ensemble des valeurs de cellules, espaces de bord retirés.
    """
    cells: set[str] = set()
    for table in payload.get("tables", ()):
        for row in table.get("ann", ()):
            cells.update(c.strip() for c in row)
        for result in table.get("methods", {}).values():
            for row in result.get("pred", ()):
                cells.update(c.strip() for c in row)
    return cells


def is_vocabulary(label: str) -> bool:
    """Indique si tous les mots significatifs du libellé sont du vocabulaire comptable.

    Un libellé sans aucun mot significatif — « (50 % au moins) », composé de chiffres et
    de mots de liaison — est accepté : il ne peut désigner aucune entité.

    Args:
        label: libellé conservé en clair.

    Returns:
        True si le libellé est structurel et peut légitimement rester lisible.
    """
    words = [w for w in _norm_label(label).split() if w and not w.isdigit()]
    significant = [w for w in words if w not in _STOPWORDS]
    return all(w in _VOCAB for w in significant)


def check(payload: dict) -> list[str]:
    """Applique tous les contrôles d'anonymisation.

    Args:
        payload: contenu de comparaisons.json.

    Returns:
        Liste des violations, vide si le jeu de données est conforme.
    """
    problems: list[str] = []

    ids = [t.get("id", "") for t in payload.get("tables", ())]
    for table_id in ids:
        if not _TABLE_ID_RE.match(table_id):
            problems.append(f"identifiant de tableau non neutre : {table_id!r}")

    cells = collect_cells(payload)
    for cell in sorted(cells):
        if not cell or _PSEUDONYM_RE.match(cell) or _NON_ENTITY_RE.match(cell):
            continue
        if not _TEXTUAL_RE.search(cell):
            continue
        if not is_vocabulary(cell):
            problems.append(f"libellé en clair hors vocabulaire : {cell[:90]!r}")
        if _SIREN_RE.search(cell):
            problems.append(f"suite de 9 chiffres dans un libellé : {cell[:90]!r}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DATA_PATH, help="JSON à vérifier")
    parser.add_argument(
        "--list", action="store_true", help="lister les libellés conservés en clair"
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"[ERREUR] {args.path} absent — lancer d'abord build_data.py", file=sys.stderr)
        return 2

    payload = json.loads(args.path.read_text(encoding="utf-8"))

    if args.list:
        kept = sorted(
            c
            for c in collect_cells(payload)
            if c and _TEXTUAL_RE.search(c) and not _PSEUDONYM_RE.match(c)
        )
        print(f"{len(kept)} libellé(s) conservé(s) en clair :")
        for label in kept:
            print("  ", label.replace("\n", " ")[:100])
        return 0

    problems = check(payload)
    n_tables = len(payload.get("tables", ()))
    if problems:
        print(f"[ÉCHEC] {len(problems)} violation(s) sur {n_tables} tableaux :", file=sys.stderr)
        for problem in problems[:40]:
            print("  -", problem, file=sys.stderr)
        if len(problems) > 40:
            print(f"  … et {len(problems) - 40} autre(s)", file=sys.stderr)
        print(
            "\nNe pas étendre _VOCAB pour faire passer ce test : vérifier pourquoi la "
            "règle a conservé ces libellés.",
            file=sys.stderr,
        )
        return 1

    print(f"[OK] {n_tables} tableaux — aucune identité en clair détectée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
