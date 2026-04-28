#!/usr/bin/env python3
"""
Pipeline d'évaluation pour l'extraction de tableaux depuis images.

Compare les tableaux prédits (CSV) produits par Marker et OpenDataLoader
avec les annotations de référence (XLSX).

Sources CSV :
  s3://projet-extraction-tableaux/reprise/output_csv/marker/
  s3://projet-extraction-tableaux/reprise/output_csv/opendataloader/
Annotations :
  s3://projet-extraction-tableaux/annotations/clean/
Résultats :
  s3://projet-extraction-tableaux/reprise/eval/evaluation.parquet

Métriques (type rappel, une ligne par couple fichier × méthode) :
  col_recovery      – taux de colonnes de l'annotation retrouvées
  row_recovery      – taux de lignes de l'annotation retrouvées
  numeric_recovery  – taux de cellules numériques (hors en-têtes) bien récupérées
  total_extraction  – 1 si toutes les cellules non-vides sont matchées exactement

Usage :
    uv run evaluation_extraction.py [--threshold 0.5]
"""

import io
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extraction_common.s3 import get_s3_fs

BUCKET = "projet-extraction-tableaux"
S3_ANNOTATIONS = f"{BUCKET}/annotations/clean"
S3_EVAL_OUTPUT = f"{BUCKET}/reprise/eval/evaluation.parquet"

METHODS: dict[str, str] = {
    "marker":         f"{BUCKET}/reprise/output_csv/marker",
    "opendataloader": f"{BUCKET}/reprise/output_csv/opendataloader",
    "chandra":        f"{BUCKET}/reprise/output_csv/chandra",
}


# ── Chargement S3 ─────────────────────────────────────────────────────────────

def _load_csv(fs, path: str) -> pd.DataFrame:
    with fs.open(path, "r", encoding="utf-8-sig") as f:
        return pd.read_csv(f, header=None, dtype=str, sep=None, engine="python").fillna("")


def _load_xlsx(fs, path: str) -> pd.DataFrame:
    with fs.open(path, "rb") as f:
        return pd.read_excel(io.BytesIO(f.read()), header=None, dtype=str).fillna("")


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


# ── Helpers cellules ──────────────────────────────────────────────────────────

def _is_numeric(value: str) -> bool:
    s = value.strip().replace(",", ".").replace(" ", "").replace(" ", "")
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_empty(value: str) -> bool:
    return value.strip() == ""


def _non_numeric_rate(series: pd.Series) -> float:
    non_empty = [v for v in series if not _is_empty(v)]
    if not non_empty:
        return 1.0
    return 1.0 - sum(1 for v in non_empty if _is_numeric(v)) / len(non_empty)


# ── Étape 1 : Détection des en-têtes ─────────────────────────────────────────

def detect_column_header_height(df: pd.DataFrame) -> int:
    """
    Nombre de lignes formant l'en-tête des colonnes.
    Intègre les lignes numériques initiales, puis les lignes majoritairement
    non-numériques (taux non-numérique >= 0.5).
    """
    n, i = len(df), 0
    while i < n and _non_numeric_rate(df.iloc[i]) < 0.5:
        i += 1
    while i < n and _non_numeric_rate(df.iloc[i]) >= 0.5:
        i += 1
    return i


def detect_row_header_width(df: pd.DataFrame) -> int:
    """
    Nombre de colonnes formant l'en-tête des lignes.
    Première colonne toujours incluse, puis les suivantes tant qu'elles
    contiennent au moins une cellule numérique.
    """
    n_cols = len(df.columns)
    if n_cols == 0:
        return 0
    width = 1
    for c in range(1, n_cols):
        if not any(_is_numeric(v) for v in df.iloc[:, c]):
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
) -> dict:
    """
    Calcule les métriques de rappel pour une paire (annotation, prédiction).
    Les valeurs sont comparées en tant que chaînes de caractères (.strip()).
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
    total_num = recovered_num = 0
    for r in range(ann_hrows, n_ann_rows):
        for c in range(ann_hcols, n_ann_cols):
            val = annotation.iloc[r, c]
            if not _is_numeric(val):
                continue
            total_num += 1
            if r in row_match and c in col_match:
                pr, pc = row_match[r], col_match[c]
                if pr < len(prediction) and pc < len(prediction.columns):
                    if prediction.iloc[pr, pc].strip() == val.strip():
                        recovered_num += 1

    numeric_recovery = recovered_num / total_num if total_num else 1.0

    # Indicatrice d'extraction totale
    total_ok = True
    for r in range(n_ann_rows):
        if not total_ok:
            break
        for c in range(n_ann_cols):
            val = annotation.iloc[r, c]
            if _is_empty(val):
                continue
            if r not in row_match or c not in col_match:
                total_ok = False
                break
            pr, pc = row_match[r], col_match[c]
            if pr >= len(prediction) or pc >= len(prediction.columns):
                total_ok = False
                break
            if prediction.iloc[pr, pc].strip() != val.strip():
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


# ── Évaluation par lot ────────────────────────────────────────────────────────

def evaluate_dataset(threshold: float = 0.5) -> pd.DataFrame:
    fs = get_s3_fs()
    all_results: list[dict] = []

    for method, pred_prefix in METHODS.items():
        pairs = _list_pairs(fs, pred_prefix)
        print(f"\n[{method}] {len(pairs)} paire(s) trouvée(s)")

        for name, ann_path, pred_path in pairs:
            try:
                ann_df = _load_xlsx(fs, ann_path)
                pred_df = _load_csv(fs, pred_path)
                metrics = evaluate_pair(ann_df, pred_df, threshold=threshold)
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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation de l'extraction de tableaux")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Seuil de similarité Levenshtein pour le matching (défaut : 0.5)",
    )
    args = parser.parse_args()
    evaluate_dataset(threshold=args.threshold)
