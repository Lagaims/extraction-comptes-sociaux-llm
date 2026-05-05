#!/usr/bin/env python3
"""
Pipeline d'évaluation pour l'extraction de tableaux depuis images.

Compare les tableaux prédits (CSV) produits par Marker et OpenDataLoader
avec les annotations de référence (XLSX).

Sources CSV :
  s3://projet-extraction-tableaux/reprise/output_csv/marker/
  s3://projet-extraction-tableaux/reprise/output_csv/opendataloader/
  s3://projet-extraction-tableaux/reprise/output_csv/chandra/
Annotations :
  s3://projet-extraction-tableaux/annotations/clean/
Résultats :
  s3://projet-extraction-tableaux/reprise/eval/evaluation.parquet

Métriques (type rappel, une ligne par couple fichier × méthode) :
  col_recovery      – taux de colonnes de l'annotation retrouvées
  row_recovery      – taux de lignes de l'annotation retrouvées
  numeric_recovery  – taux de cellules numériques (hors en-têtes) bien récupérées
  total_extraction  – 1 si structure complète (toutes colonnes/lignes matchées) et toutes
                      les cellules numériques de la zone de données sont récupérées
                      (tolérance : espaces dans les nombres, pourcentages décimaux/pourcentage)

Usage :
    uv run evaluation_extraction.py [--threshold 0.5 --cell-delta 0]
"""

import io
import re
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from extraction_common.s3 import get_s3_fs

BUCKET = "projet-extraction-tableaux"
S3_ANNOTATIONS = f"{BUCKET}/annotations/clean"
S3_EVAL_OUTPUT = f"{BUCKET}/reprise/eval/evaluation.parquet"
S3_CORRESPONDANCES = f"{BUCKET}/reprise/correspondances.parquet"

_SIREN_RE = re.compile(r"\d{9}")

METHODS: dict[str, str] = {
    "marker":           f"{BUCKET}/reprise/output_csv/marker",
    "opendataloader":   f"{BUCKET}/reprise/output_csv/opendataloader",
    "chandra":          f"{BUCKET}/reprise/output_csv/chandra",
    "marker_last_work": f"{BUCKET}/LLM_eval/output_csv/marker_last_work",
}


# ── Chargement S3 ─────────────────────────────────────────────────────────────

def _load_correspondances(fs) -> dict[str, list[str]]:
    """
    Charge le parquet de correspondances.
    Retourne {pure_siren: [xlsx_path_1, ...]} (chemins s3fs sans schéma s3://).
    Le SIREN pur (9 chiffres) est extrait du stem PDF (colonne 'siren').
    """
    with fs.open(S3_CORRESPONDANCES, "rb") as f:
        df = pd.read_parquet(f)
    mapping: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        m = _SIREN_RE.search(str(row["siren"]))
        if not m:
            continue
        try:
            xlsx_paths = sorted(p.removeprefix("s3://") for p in row["xlsx"])
        except TypeError:
            continue
        if xlsx_paths:
            mapping[m.group()] = xlsx_paths
    return mapping


def _load_csv(fs, path: str) -> pd.DataFrame:
    import csv as _csv
    with fs.open(path, "r", encoding="utf-8-sig") as f:
        rows = list(_csv.reader(f, delimiter=";"))
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    return pd.DataFrame(padded, dtype=str).fillna("")


def _load_xlsx(fs, path: str) -> pd.DataFrame:
    with fs.open(path, "rb") as f:
        df = pd.read_excel(io.BytesIO(f.read()), header=None, dtype=str).fillna("")
    mask = df.apply(lambda row: row.str.strip().eq("").all(), axis=1)
    return df[~mask].reset_index(drop=True)


def _list_pairs(fs, pred_prefix: str) -> list[tuple[str, str, str]]:
    """
    Apparie annotations (.xlsx) et prédictions (.csv) par stem de fichier.
    Retourne une liste de (nom, annotation_path, prediction_path).
    """
    ann = {Path(p).stem: p for p in fs.glob(f"{S3_ANNOTATIONS}/*.xlsx")}
    pred = {Path(p).stem: p for p in fs.glob(f"{pred_prefix}/*.csv")}

    only_ann = sorted(set(ann) - set(pred))
    only_pred = sorted(set(pred) - set(ann))
    if only_ann:
        print(f"    [WARN] {len(only_ann)} annotation(s) sans prédiction.")
    if only_pred:
        print(f"    [WARN] {len(only_pred)} prédiction(s) sans annotation.")

    return [(name, ann[name], pred[name]) for name in sorted(set(ann) & set(pred))]


def _list_pairs_from_correspondances(fs, pred_prefix: str) -> list[tuple[str, str, str]]:
    """
    Apparie annotations et prédictions via le parquet de correspondances.
    La i-ème annotation (triée) d'un SIREN est appariée à la prédiction {siren}_{i}.csv.
    Retourne une liste de (nom, annotation_path, prediction_path).
    """
    correspondances = _load_correspondances(fs)
    pred = {Path(p).stem: p for p in fs.glob(f"{pred_prefix}/*.csv")}

    pairs = []
    matched_preds: set[str] = set()
    for pure_siren, xlsx_paths in correspondances.items():
        for rank, xlsx_path in enumerate(xlsx_paths, start=1):
            stem = f"{pure_siren}_{rank}"
            if stem in pred:
                pairs.append((stem, xlsx_path, pred[stem]))
                matched_preds.add(stem)

    unmatched = sorted(set(pred) - matched_preds)
    if unmatched:
        print(f"    [WARN] {len(unmatched)} prédiction(s) sans annotation.")

    return sorted(pairs)


# ── Helpers cellules ──────────────────────────────────────────────────────────

def _is_numeric(value: str) -> bool:
    s = value.strip().replace(",", ".").replace(" ", "").replace(" ", "")
    if not s:
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_empty(value: str) -> bool:
    return value.strip() == ""


_NUMERIC_PLACEHOLDERS = {
    "-", "–", "—", "n.a.", "n/a", "nd", "n.d.", "ns", "nc", "n.c.",
}
_UNIT_SUFFIX_RE = re.compile(
    r'(?i)\s*(€|eur|euros?|usd|\$|gbp|£|nok|sek|chf|jpy|¥|kr|%|pp|bps?)\s*$'
)


def _looks_numeric(value: str) -> bool:
    """Variante permissive de `_is_numeric`, dédiée à la détection d'en-têtes.

    Accepte en plus :
    - parenthèses comptables : (14) → -14
    - unités en suffixe       : 15,24 €, 344 369 NOK, 100,00%
    - placeholders d'absence  : -, NC, ND, N/A
    """
    s = value.strip()
    if not s:
        return True
    if s.lower() in _NUMERIC_PLACEHOLDERS:
        return True
    s = _UNIT_SUFFIX_RE.sub("", s).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = s.replace(",", ".").replace(" ", "").replace(" ", "")
    if not s:
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def _normalize_numeric_str(val: str) -> str:
    """Normalise un nombre textuel pour la comparaison.

    - Supprime les espaces séparateurs de milliers : '25 000' → '25000'.
    - Convertit les pourcentages en décimales : '100%' → '1', '66,67%' → '0.6667'.
    """
    s = val.strip()
    # Espaces séparateurs de milliers (normaux, insécables  ,  )
    s = re.sub(r'(\d)[\s  ]+(\d)', r'\1\2', s)
    # Pourcentages → décimales
    m = re.fullmatch(r'(-?\d+(?:[,\.]\d+)?)\s*%', s)
    if m:
        try:
            n = float(m.group(1).replace(',', '.')) / 100
            s = f'{n:.6g}'
        except ValueError:
            pass
    return s


def _cell_recovered(
    prediction: pd.DataFrame, pr: int, pc: int, val: str, delta: int, strict: bool = False
) -> bool:
    """Retourne True si val est trouvée dans prediction à (pr, pc±delta).

    Si `strict` est False (défaut), comparaison via `_normalize_numeric_str`
    (tolérante aux espaces séparateurs et au format des pourcentages).
    Si `strict` est True, comparaison de chaîne brute après strip — utilisée
    pour les cellules avec unités, parenthèses ou placeholders.
    """
    target = val.strip() if strict else _normalize_numeric_str(val)
    for dc in range(-delta, delta + 1):
        c = pc + dc
        if 0 <= c < len(prediction.columns):
            cand = prediction.iloc[pr, c]
            cand_norm = cand.strip() if strict else _normalize_numeric_str(cand)
            if cand_norm == target:
                return True
    return False


def _non_numeric_rate(series: pd.Series) -> float:
    non_empty = [v for v in series if not _is_empty(v)]
    if not non_empty:
        return 1.0
    return 1.0 - sum(1 for v in non_empty if _looks_numeric(v)) / len(non_empty)


# ── Étape 1 : Détection des en-têtes ─────────────────────────────────────────

def detect_column_header_height(df: pd.DataFrame) -> int:
    """
    Nombre de lignes formant l'en-tête des colonnes.
    Intègre les lignes numériques initiales, puis les lignes majoritairement
    non-numériques (taux non-numérique >= 0.5).

    Garde-fous :
    - Si aucune ligne textuelle n'est absorbée par la phase 2, on considère qu'il
      n'y a pas d'en-tête de colonnes (ex. tableaux filiales sans ligne d'en-tête).
    - Si la phase 2 absorbe TOUTES les lignes restantes jusqu'à la fin (typique
      des tableaux text-heavy où chaque ligne a `rate >= 0.5` sans pour autant
      être un en-tête), on retombe sur les seules lignes 100% non-numériques.
    """
    n, i = len(df), 0
    while i < n and _non_numeric_rate(df.iloc[i]) < 0.5:
        i += 1
    phase1_end = i
    while i < n and _non_numeric_rate(df.iloc[i]) >= 0.5:
        i += 1
    if i == phase1_end:
        return 0
    if i == n:
        j = phase1_end
        while j < n and _non_numeric_rate(df.iloc[j]) >= 1.0:
            j += 1
        return j
    return i


def detect_row_header_width(df: pd.DataFrame) -> int:
    """
    Nombre de colonnes formant l'en-tête des lignes.
    Première colonne toujours incluse, puis les suivantes tant qu'elles
    contiennent au moins une cellule numérique stricte ET sont
    majoritairement non-numériques (taux non-numérique > 0.5).
    """
    n_cols = len(df.columns)
    if n_cols == 0:
        return 0
    width = 1
    for c in range(1, n_cols):
        col = df.iloc[:, c]
        if not any(_looks_numeric(v) and not _is_empty(v) for v in col):
            break
        if _non_numeric_rate(col) <= 0.5:
            break
        width += 1
    return width


# ── Étape 2 : Matching Levenshtein + Gale-Shapley ────────────────────────────

def _levenshtein_distance(s: str, t: str) -> int:
    m, n = len(s), len(t)
    if m < n:
        s, t, m, n = t, s, n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def _lev_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - _levenshtein_distance(a, b) / max(len(a), len(b))


def _build_header_texts(
    df: pd.DataFrame, axis: str, n_header_rows: int, n_header_cols: int
) -> list[str]:
    """
    Texte représentatif de chaque colonne (axis='col') ou ligne (axis='row'),
    formé en concaténant les cellules de l'en-tête correspondant.
    """
    texts = []
    if axis == "col":
        for c in range(len(df.columns)):
            parts = [df.iloc[r, c] for r in range(n_header_rows) if not _is_empty(df.iloc[r, c])]
            texts.append(" | ".join(parts))
    else:
        for r in range(len(df)):
            parts = [df.iloc[r, c] for c in range(n_header_cols) if not _is_empty(df.iloc[r, c])]
            texts.append(" | ".join(parts))
    return texts


def _gale_shapley(scores: np.ndarray) -> dict[int, int]:
    """
    Matching stable (Gale-Shapley) entre proposeurs (annotation) et accepteurs (prédiction).
    Retourne {ann_idx: pred_idx}.
    """
    n_a, n_b = scores.shape
    prefs_a = [list(np.argsort(-scores[i])) for i in range(n_a)]
    rank_b = {
        j: {i: r for r, i in enumerate(np.argsort(-scores[:, j]))}
        for j in range(n_b)
    }
    free_a = list(range(n_a))
    next_prop = [0] * n_a
    match_b: dict[int, int] = {}
    match_a: dict[int, int] = {}

    while free_a:
        a = free_a.pop(0)
        if next_prop[a] >= n_b:
            continue
        b = prefs_a[a][next_prop[a]]
        next_prop[a] += 1
        if b not in match_b:
            match_b[b] = a
            match_a[a] = b
        else:
            cur = match_b[b]
            if rank_b[b].get(a, n_a) < rank_b[b].get(cur, n_a):
                match_b[b] = a
                match_a[a] = b
                del match_a[cur]
                free_a.append(cur)
            else:
                free_a.append(a)
    return match_a


def _match_headers(
    ann_texts: list[str], pred_texts: list[str], threshold: float
) -> dict[int, int]:
    """Retourne {ann_idx: pred_idx} pour les paires dont la similarité >= threshold."""
    n_a, n_p = len(ann_texts), len(pred_texts)
    if n_a == 0 or n_p == 0:
        return {}
    scores = np.array([
        [_lev_similarity(ann_texts[i], pred_texts[j]) for j in range(n_p)]
        for i in range(n_a)
    ])
    raw = _gale_shapley(scores)
    return {i: j for i, j in raw.items() if scores[i, j] >= threshold}


# ── Étape 3 : Métriques ───────────────────────────────────────────────────────

def evaluate_pair(
    annotation: pd.DataFrame,
    prediction: pd.DataFrame,
    threshold: float = 0.5,
    cell_delta: int = 0,
) -> dict:
    """
    Calcule les métriques de rappel pour une paire (annotation, prédiction).
    Les valeurs sont comparées en tant que chaînes de caractères (.strip()).
    cell_delta : tolérance en nombre de colonnes (±) pour total_extraction uniquement.
    """
    ann_hrows = detect_column_header_height(annotation)
    ann_hcols = detect_row_header_width(annotation)
    pred_hrows = detect_column_header_height(prediction)
    pred_hcols = detect_row_header_width(prediction)

    col_match = _match_headers(
        _build_header_texts(annotation, "col", ann_hrows, ann_hcols),
        _build_header_texts(prediction, "col", pred_hrows, pred_hcols),
        threshold,
    )
    row_match = _match_headers(
        _build_header_texts(annotation, "row", ann_hrows, ann_hcols),
        _build_header_texts(prediction, "row", pred_hrows, pred_hcols),
        threshold,
    )

    n_ann_rows, n_ann_cols = len(annotation), len(annotation.columns)
    col_recovery = len(col_match) / n_ann_cols if n_ann_cols else 0.0
    row_recovery = len(row_match) / n_ann_rows if n_ann_rows else 0.0

    # Cellules numériques (zone de données, hors en-têtes)
    # On compte toutes les cellules `_looks_numeric` (nombres purs, unités, parenthèses,
    # placeholders). La comparaison avec la prédiction est :
    #   - normalisée (espaces, %)   pour les nombres purs (`_is_numeric` True)
    #   - stricte (chaîne identique) pour les cellules avec unités / parenthèses /
    #     placeholders, où l'on évalue l'exactitude des caractères extraits
    total_num = recovered_num = 0
    for r in range(ann_hrows, n_ann_rows):
        for c in range(ann_hcols, n_ann_cols):
            val = annotation.iloc[r, c]
            if not _looks_numeric(val):
                continue
            total_num += 1
            if r in row_match and c in col_match:
                pr, pc = row_match[r], col_match[c]
                if pr < len(prediction) and pc < len(prediction.columns):
                    pred_val = prediction.iloc[pr, pc]
                    if _is_numeric(val):
                        match = _normalize_numeric_str(pred_val) == _normalize_numeric_str(val)
                    else:
                        match = pred_val.strip() == val.strip()
                    if match:
                        recovered_num += 1

    numeric_recovery = recovered_num / total_num if total_num else float("nan") # cas sans cellules 

    # Indicatrice d'extraction totale :
    # (1) structure complète : toutes les colonnes ET lignes de l'annotation sont matchées
    # (2) toutes les cellules numériques de la zone de données sont récupérées (normalisées)
    total_ok = (len(col_match) == n_ann_cols and len(row_match) == n_ann_rows)
    if total_ok:
        for r in range(ann_hrows, n_ann_rows):
            if not total_ok:
                break
            for c in range(ann_hcols, n_ann_cols):
                val = annotation.iloc[r, c]
                if not _looks_numeric(val):
                    continue
                if r not in row_match or c not in col_match:
                    total_ok = False
                    break
                pr, pc = row_match[r], col_match[c]
                if pr >= len(prediction) or pc >= len(prediction.columns):
                    total_ok = False
                    break
                strict = not _is_numeric(val)
                if not _cell_recovered(prediction, pr, pc, val, cell_delta, strict=strict):
                    total_ok = False
                    break

    return {
        "col_recovery": col_recovery,
        "row_recovery": row_recovery,
        "numeric_recovery": numeric_recovery,
        "total_extraction": int(total_ok),
        "n_ann_rows": n_ann_rows,
        "n_ann_cols": n_ann_cols,
        "n_matched_rows": len(row_match),
        "n_matched_cols": len(col_match),
        "n_numeric_cells": total_num,
        "n_recovered_numeric": recovered_num,
        "ann_header_rows": ann_hrows,
        "ann_header_cols": ann_hcols,
    }


# ── Comptage de tableaux par SIREN ────────────────────────────────────────────

_BASE_STEM_RE = re.compile(r'^(.*?)_(\d+)$')


def _base_stem(name: str) -> str:
    """Retire le suffixe _N d'un nom de fichier pour obtenir l'identifiant SIREN."""
    m = _BASE_STEM_RE.match(name)
    return m.group(1) if m else name


def _count_per_base(fs, prefix: str, ext: str) -> Counter:
    """Compte le nombre de fichiers par SIREN (base_stem) dans un dossier S3."""
    stems = [Path(p).stem for p in fs.glob(f"{prefix}/*{ext}")]
    return Counter(_base_stem(s) for s in stems)


# ── Évaluation par lot ────────────────────────────────────────────────────────

def evaluate_dataset(threshold: float = 0.5, cell_delta: int = 0) -> pd.DataFrame:
    fs = get_s3_fs()
    all_results: list[dict] = []

    ann_counts_default = _count_per_base(fs, S3_ANNOTATIONS, ".xlsx")

    for method, pred_prefix in METHODS.items():
        if method == "marker_last_work":
            pairs = _list_pairs_from_correspondances(fs, pred_prefix)
            correspondances = _load_correspondances(fs)
            ann_counts = Counter({siren: len(paths) for siren, paths in correspondances.items()})
        else:
            pairs = _list_pairs(fs, pred_prefix)
            ann_counts = ann_counts_default
        print(f"\n[{method}] {len(pairs)} paire(s) trouvée(s)")

        pred_counts = _count_per_base(fs, pred_prefix, ".csv")

        for name, ann_path, pred_path in pairs:
            base = _base_stem(name)
            try:
                ann_df = _load_xlsx(fs, ann_path)
                pred_df = _load_csv(fs, pred_path)
                metrics = evaluate_pair(ann_df, pred_df, threshold=threshold, cell_delta=cell_delta)
                metrics.update({"fichier": name, "methode": method})
                print(
                    f"  {name:<30} "
                    f"col={metrics['col_recovery']:.3f}  "
                    f"row={metrics['row_recovery']:.3f}  "
                    f"num={metrics['numeric_recovery']:.3f}  "
                    f"total={metrics['total_extraction']}"
                )
            except Exception as e:
                print(f"  [ERR] {name}: {e}")
                metrics = {
                    "fichier": name,
                    "methode": method,
                    "col_recovery": None,
                    "row_recovery": None,
                    "numeric_recovery": None,
                    "total_extraction": None,
                }
            metrics["n_ann_tables"] = ann_counts.get(base, 0)
            metrics["n_pred_tables"] = pred_counts.get(base, 0)
            all_results.append(metrics)

    df = pd.DataFrame(all_results)
    _save_parquet(fs, df)
    _print_summary(df)
    return df


def _save_parquet(fs, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    fs.pipe(S3_EVAL_OUTPUT, buf.getvalue())
    print(f"\nRésultats sauvegardés : s3://{S3_EVAL_OUTPUT}")


def _print_summary(df: pd.DataFrame) -> None:
    metric_cols = ["col_recovery", "row_recovery", "numeric_recovery", "total_extraction"]
    print("\n=== Moyennes par méthode ===")
    for method in df["methode"].dropna().unique():
        sub = df[df["methode"] == method]
        print(f"\n  {method}")
        for col in metric_cols:
            print(f"    {col:<25}: {sub[col].dropna().mean():.4f}")

        if "n_ann_tables" in sub.columns and "n_pred_tables" in sub.columns:
            siren_df = sub.copy()
            siren_df["_siren"] = siren_df["fichier"].apply(_base_stem)
            siren_df = siren_df.drop_duplicates("_siren")
            n_total = len(siren_df)
            n_match = (siren_df["n_pred_tables"] == siren_df["n_ann_tables"]).sum()
            print(f"    {'---':<25}")
            print(f"    {'table_count_accuracy':<25}: {n_match / n_total:.4f}  ({n_match}/{n_total} SIREN)")
            print(f"    {'moy. tableaux annotés':<25}: {siren_df['n_ann_tables'].mean():.2f}")
            print(f"    {'moy. tableaux détectés':<25}: {siren_df['n_pred_tables'].mean():.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation de l'extraction de tableaux")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Seuil de similarité Levenshtein pour le matching (défaut : 0.5)",
    )
    parser.add_argument(
        "--cell-delta", type=int, default=0,
        help="Tolérance en colonnes (±) pour numeric_recovery et total_extraction (défaut : 0)",
    )
    args = parser.parse_args()
    evaluate_dataset(threshold=args.threshold, cell_delta=args.cell_delta)
