"""
API d'extraction de tableaux via le VLM Chandra (vllm, compatible OpenAI).

Chaque page du PDF est convertie en image puis envoyée au VLM avec un prompt
demandant d'extraire le tableau sous forme JSON structuré.

Lancement :
    uv run uvicorn main_chandra:app --host 0.0.0.0 --port 8003 --app-dir src

Variables d'environnement :
    CHANDRA_BASE_URL   URL du serveur vllm  (défaut: https://projet-models-hf-vllm.user.lab.sspcloud.fr/v1)
    CHANDRA_MODEL      Nom du modèle        (défaut: datalab-to/chandra-ocr-2)
    CHANDRA_API_KEY    Clé API              (défaut: "", convention vllm)
    CHANDRA_DPI        Résolution PDF→image (défaut: 200)
"""

import base64
import json
import os
import re
import shutil
import tempfile

import pymupdf as fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

load_dotenv()

os.environ.setdefault("CHANDRA_BASE_URL", "https://projet-models-hf-vllm.user.lab.sspcloud.fr/v1")
os.environ.setdefault("CHANDRA_MODEL", "datalab-to/chandra-ocr-2")
os.environ.setdefault("CHANDRA_API_KEY", "")
os.environ.setdefault("CHANDRA_DPI", "200")

EXTRACTION_PROMPT = """\
Look at this image and extract every table you can see.
Return a JSON object with this exact structure — no markdown, no explanation, only JSON:
{
  "tables": [
    {
      "rows": [
        ["cell_1_1", "cell_1_2", "..."],
        ["cell_2_1", "cell_2_2", "..."]
      ]
    }
  ]
}
Rules:
- Include header rows as the first row(s) of each table.
- Use empty string "" for empty cells.
- If no table is visible, return {"tables": []}.
"""

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
    """Convertit chaque page du PDF en image PNG encodée en base64."""
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
    doc.close()
    return images


# ── Appel VLM ────────────────────────────────────────────────────────────────

async def _extract_tables_from_image(
    client: AsyncOpenAI, b64_image: str, model: str
) -> list[list[list[str]]]:
    """Envoie une image au VLM et retourne les tableaux extraits."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
    )
    return _parse_vlm_response(response.choices[0].message.content or "")


def _parse_vlm_response(content: str) -> list[list[list[str]]]:
    """
    Parse la réponse du VLM en liste de tableaux.
    Gère les réponses enveloppées dans des blocs markdown (```json ... ```).
    """
    # Supprimer les blocs markdown éventuels
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"```", "", content).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Tentative de récupération : chercher un objet JSON dans la réponse
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    tables = []
    for t in data.get("tables", []):
        rows = t.get("rows", [])
        if rows:
            tables.append([[str(cell) for cell in row] for row in rows])
    return tables


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
                tables = await _extract_tables_from_image(client, b64_image, model)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur VLM page {page_num} : {e}",
                )
            pages.append({"page": page_num, "tables": tables})

    await client.close()
    return JSONResponse(content={"pages": pages})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
