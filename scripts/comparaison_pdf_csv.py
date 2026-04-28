import pandas as pd
from pathlib import Path

from extraction_common.s3 import get_s3_fs

path_xlsx = "s3://projet-extraction-tableaux/annotations/clean/*.xlsx"
path_pdf = "s3://projet-extraction-tableaux/pdf/tableaux_representatifs/*.pdf"

fs = get_s3_fs()

xlsx_files = fs.glob(path_xlsx)
pdf_files = fs.glob(path_pdf)


# Suppression des suffixes _1, _2, ... pour retrouver le SIREN de base
def siren_base(stem):
    import re
    return re.sub(r"_\d+$", "", stem)


# Regroupement des xlsx par SIREN de base (plusieurs fichiers possibles)
xlsx_par_siren = {}
for f in xlsx_files:
    siren = siren_base(Path(f).stem)
    xlsx_par_siren.setdefault(siren, []).append(f"s3://{f}")

# Tri pour un ordre stable
for siren in xlsx_par_siren:
    xlsx_par_siren[siren].sort()

pdf_par_siren = {Path(f).stem: f"s3://{f}" for f in pdf_files}

tous_les_sirens = set(xlsx_par_siren.keys()) | set(pdf_par_siren.keys())

correspondances = {
    siren: {
        "pdf": pdf_par_siren.get(siren),
        "xlsx": xlsx_par_siren.get(siren),
    }
    for siren in sorted(tous_les_sirens)
}

df = pd.DataFrame([
    {"siren": siren, "pdf": vals["pdf"], "xlsx": vals["xlsx"]}
    for siren, vals in correspondances.items()
])

output_path = "s3://projet-extraction-tableaux/reprise/correspondances.parquet"
print(f"Nombre de siren détectés : {len(df)}")
print(f"Nombre de pdf détectés : {sum(1 for v in correspondances.values() if v["pdf"] is not None)}")
print(f"Nombre de xlsx d'annotations détectés : {sum(len(v["xlsx"]) for v in correspondances.values() if v["xlsx"] is not None)}")

with fs.open(output_path, "wb") as f:
    df.to_parquet(f, index=False)
