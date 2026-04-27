import os
import s3fs
import pandas as pd
from pathlib import Path

path_xlsx = "s3://projet-extraction-tableaux/annotations/clean/*.xlsx"
path_pdf = "s3://projet-extraction-tableaux/pdf/tableaux_representatifs/*.pdf"

fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    token=os.environ.get("AWS_SESSION_TOKEN"),
    client_kwargs={"endpoint_url": "https://"+os.environ["AWS_S3_ENDPOINT"]},
)

print(fs.ls(""))

xlsx_files = fs.glob(path_xlsx)
pdf_files = fs.glob(path_pdf)

# Indexation par SIRET (nom de base sans extension)
xlsx_par_siret = {Path(f).stem: f"s3://{f}" for f in xlsx_files}
pdf_par_siret = {Path(f).stem: f"s3://{f}" for f in pdf_files}

tous_les_sirets = set(xlsx_par_siret.keys()) | set(pdf_par_siret.keys())

correspondances = {
    siret: {
        "pdf": pdf_par_siret.get(siret),
        "xlsx": xlsx_par_siret.get(siret),
    }
    for siret in sorted(tous_les_sirets)
}

df = pd.DataFrame([
    {"siret": siret, "pdf": vals["pdf"], "xlsx": vals["xlsx"]}
    for siret, vals in correspondances.items()
])

output_path = "s3://projet-extraction-tableaux/reprise/correspondances.parquet"
with fs.open(output_path, "wb") as f:
    df.to_parquet(f, index=False)
