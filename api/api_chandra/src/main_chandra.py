"""
API d'extraction de tableaux via le VLM Chandra (vllm, compatible OpenAI).

Chaque page du PDF est convertie en image puis envoyée au VLM.
Chandra retourne du HTML natif (<table>), parsé ensuite en JSON structuré.

Lancement :
    uv run uvicorn main_chandra:app --host 0.0.0.0 --port 8003 --app-dir src

Variables d'environnement :
    CHANDRA_BASE_URL    URL du serveur vllm  (défaut: https://projet-models-hf-vllm.user.lab.sspcloud.fr/v1)
    CHANDRA_MODEL       Nom du modèle        (défaut: datalab-to/chandra-ocr-2)
    CHANDRA_API_KEY     Clé API              (défaut: EMPTY, convention vllm)
    CHANDRA_DPI         Résolution PDF→image (défaut: 200)
    CHANDRA_RETRIES     Tentatives par page  (défaut: 3)
    CHANDRA_RETRY_DELAY Délai initial (s)    (défaut: 2, multiplié par le numéro de tentative)
"""

import asyncio
import base64
import os
import shutil
import tempfile
from html.parser import HTMLParser

import pymupdf as fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

load_dotenv()

os.environ.setdefault("CHANDRA_BASE_URL", "https://projet-models-hf-vllm.user.lab.sspcloud.fr/v1")
os.environ.setdefault("CHANDRA_MODEL", "datalab-to/chandra-ocr-2")
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


# ── Parsing HTML → tableaux ───────────────────────────────────────────────────

class _TableParser(HTMLParser):
    """Extrait les tableaux HTML en listes de listes de chaînes."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._current_table = []
        elif tag in ("tr",) and self._current_table is not None:
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_cell = ""

    def handle_endtag(self, tag):
        if tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._current_cell is not None:
            self._current_row.append(self._current_cell.strip())
            self._current_cell = None

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell += data


def _parse_chandra_html(content: str) -> list[list[list[str]]]:
    parser = _TableParser()
    parser.feed(content)
    return parser.tables


# ── Appel VLM ────────────────────────────────────────────────────────────────

async def _extract_tables_from_image(
    client: AsyncOpenAI, b64_image: str, model: str
) -> list[list[list[str]]]:
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
            )
            return _parse_chandra_html(response.choices[0].message.content or "")
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
