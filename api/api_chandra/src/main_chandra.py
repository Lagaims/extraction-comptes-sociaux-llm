"""
API d'extraction de tableaux via le VLM Chandra (vllm, compatible OpenAI).

Chaque page du PDF est convertie en image puis envoyée au VLM. Chandra retourne du HTML
natif (<table>) : **c'est ce HTML qui est renvoyé tel quel**, page par page.

    {"metadata": {"model": ..., "dpi": ...},
     "pages": [{"page": 1, "html": "<table>…</table>"}]}

L'API ne structure plus la réponse. Elle le faisait auparavant, en aplatissant le HTML en
listes de listes de chaînes (`pages[].tables`), ce qui détruisait au passage tout ce que le
modèle savait de la mise en page : `colspan`, `rowspan` — donc les cellules fusionnées —
et les `<br>`, dont la disparition soudait les mots de deux lignes d'un même libellé
(« Prêts etavancesconsentispar laSociété », 48 cellules du corpus `reprise/`).

La structuration appartient à l'étape suivante, `scripts/json_to_csv.py`, qui dispose déjà
d'un parseur HTML traitant les fusions et sert le même office pour marker et
opendataloader. Conserver le HTML brut a un second avantage : aucune information n'est
perdue à l'écriture, et un changement d'avis sur la mise en forme ne demande pas de
relancer le GPU. Les JSON du format historique (`pages[].tables`) restent lus par
`json_to_csv.py`.

Lancement :
    uv run uvicorn main_chandra:app --host 0.0.0.0 --port 8003 --app-dir src

Variables d'environnement :
    CHANDRA_BASE_URL    URL du serveur vllm  (défaut: https://llm.lab.sspcloud.fr/api)
    CHANDRA_MODEL       Nom du modèle        (défaut: chandra-ocr-2)
    CHANDRA_API_KEY     Clé API              (défaut: EMPTY, convention vllm ; sur llm.lab,
                        passer REAL_LLM_API_KEY — l'endpoint est authentifié)
    CHANDRA_DPI         Résolution PDF→image (défaut: 200)
    CHANDRA_RETRIES     Tentatives par page  (défaut: 3)
    CHANDRA_RETRY_DELAY Délai initial (s)    (défaut: 2, multiplié par le numéro de tentative)

"""

import asyncio
import base64
import os
import shutil
import tempfile

import pymupdf as fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

load_dotenv()

os.environ.setdefault("CHANDRA_BASE_URL", "https://llm.lab.sspcloud.fr/api")
os.environ.setdefault("CHANDRA_MODEL", "chandra-ocr-2")
os.environ.setdefault("CHANDRA_API_KEY", "EMPTY")
os.environ.setdefault("CHANDRA_DPI", "200")
os.environ.setdefault("CHANDRA_RETRIES", "5")
os.environ.setdefault("CHANDRA_RETRY_DELAY", "2")

app = FastAPI(
    title="API Chandra PDF Extraction",
    version="1.0.0",
    description="Extraction de tableaux PDF via le VLM Chandra",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Conversion PDF → images base64 ───────────────────────────────────────────


def _pdf_to_b64_images(pdf_path: str, dpi: int) -> list[str]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return images


# ── Appel VLM ────────────────────────────────────────────────────────────────


async def _extract_html_from_image(client: AsyncOpenAI, b64_image: str, model: str) -> str:
    """Soumet une image au VLM et retourne sa réponse brute.

    Args:
        client: client OpenAI asynchrone pointant sur le serveur vllm.
        b64_image: page rendue en PNG, encodée en base64.
        model: nom du modèle servi par vllm.

    Returns:
        Le texte produit par le modèle, sans retouche : le HTML de la page, y compris ce
        qui l'entoure. Aucune structuration ici, pour ne rien perdre à l'écriture.
    """
    retries = int(os.getenv("CHANDRA_RETRIES"))
    delay = float(os.getenv("CHANDRA_RETRY_DELAY"))
    last_exc: Exception | None = None

    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))

    raise last_exc


# ── Endpoint /extract ─────────────────────────────────────────────────────────


@app.post("/extract")
async def extract(pdf: UploadFile = File(...)):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Fichier PDF requis.")

    model = os.getenv("CHANDRA_MODEL")
    dpi = int(os.getenv("CHANDRA_DPI"))
    client = AsyncOpenAI(
        base_url=os.getenv("CHANDRA_BASE_URL"),
        api_key=os.getenv("CHANDRA_API_KEY"),
        timeout=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, pdf.filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

        try:
            b64_images = _pdf_to_b64_images(pdf_path, dpi)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur conversion PDF→image : {e}")

        pages = []
        for page_num, b64_image in enumerate(b64_images, start=1):
            try:
                html = await _extract_html_from_image(client, b64_image, model)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur VLM page {page_num} : {e}",
                )
            pages.append({"page": page_num, "html": html})

    await client.close()
    # Le modèle et le DPI sont conservés avec la sortie : deux extractions du même PDF ne
    # sont comparables que si l'on sait ce qui les a produites.
    return JSONResponse(content={"metadata": {"model": model, "dpi": dpi}, "pages": pages})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
