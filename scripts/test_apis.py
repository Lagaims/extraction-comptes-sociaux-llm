#!/usr/bin/env python3
"""
Script de test pour api_marker avec des PDFs stockés dans S3.

Usage :
    # Lister les PDFs disponibles dans S3
    python scripts/test_apis.py --list

    # Tester avec un PDF spécifique
    python scripts/test_apis.py --pdf-key dossier/fichier.pdf

    # Tester avec tous les PDFs du bucket
    python scripts/test_apis.py --all

Variables d'environnement requises (ou dans un fichier .env) :
    MARKER_API_URL      URL de api_marker (défaut : http://localhost:8001)
    AWS_S3_BUCKET
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN   (optionnel)
    AWS_S3_ENDPOINT     (optionnel, ex: minio.lab.sspcloud.fr)
    AWS_REGION          (défaut : us-east-1)
"""

import argparse
import json
import os
import sys
from io import BytesIO

import requests
import s3fs
from dotenv import load_dotenv

load_dotenv()

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

MARKER_API_URL = os.getenv("MARKER_API_URL", "http://localhost:8001")


def get_s3_fs() -> s3fs.S3FileSystem:
    endpoint = AWS_S3_ENDPOINT
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return s3fs.S3FileSystem(
        key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
        token=AWS_SESSION_TOKEN,
        client_kwargs={"endpoint_url": endpoint, "region_name": AWS_REGION},
    )


def list_pdfs(fs: s3fs.S3FileSystem) -> list[str]:
    all_keys = fs.ls(AWS_S3_BUCKET, detail=False)
    return [k.removeprefix(f"{AWS_S3_BUCKET}/") for k in all_keys if k.lower().endswith(".pdf")]


def extract_pdf(fs: s3fs.S3FileSystem, pdf_key: str):
    full_path = f"{AWS_S3_BUCKET}/{pdf_key}"
    print(f"\nLecture de s3://{full_path} ...")
    with fs.open(full_path, "rb") as f:
        pdf_bytes = f.read()
    print(f"PDF lu ({len(pdf_bytes):,} octets)")

    url = f"{MARKER_API_URL.rstrip('/')}/extract"
    filename = pdf_key.split("/")[-1]
    print(f"Envoi vers {url} ...")

    response = requests.post(
        url,
        files={"pdf": (filename, BytesIO(pdf_bytes), "application/pdf")},
        timeout=300,
    )

    if response.status_code != 200:
        print(f"Erreur {response.status_code} : {response.text}")
        return None

    result = response.json()
    print(f"Résultat :\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Test api_marker avec des PDFs depuis S3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Lister les PDFs disponibles dans S3")
    group.add_argument("--pdf-key", metavar="KEY", help="Chemin du PDF dans S3 (ex: dossier/fichier.pdf)")
    group.add_argument("--all", action="store_true", help="Tester tous les PDFs du bucket")
    args = parser.parse_args()

    fs = get_s3_fs()
    pdfs = list_pdfs(fs)

    if args.list:
        if not pdfs:
            print(f"Aucun PDF trouvé dans s3://{AWS_S3_BUCKET}")
            return
        print(f"\n{len(pdfs)} PDF(s) dans s3://{AWS_S3_BUCKET} :\n")
        for k in pdfs:
            print(f"  {k}")
        return

    if args.pdf_key:
        extract_pdf(fs, args.pdf_key)
        return

    if args.all:
        if not pdfs:
            print(f"Aucun PDF trouvé dans s3://{AWS_S3_BUCKET}")
            sys.exit(1)
        print(f"\n{len(pdfs)} PDF(s) à traiter...")
        for key in pdfs:
            print(f"\n{'='*60}\n{key}\n{'='*60}")
            extract_pdf(fs, key)


if __name__ == "__main__":
    main()
