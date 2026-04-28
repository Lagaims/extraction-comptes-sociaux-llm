from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile

from dotenv import load_dotenv
from data_management.extract_image_to_json import extract_pdf, load_models, PdfConversionError, MarkerConversionError

load_dotenv()

# Valeur par défaut : marker_proxy en local
os.environ.setdefault("PROXY_URL", "http://localhost:1324/v1")

artifact_dict: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact_dict
    artifact_dict = load_models()
    yield


app = FastAPI(
    title="API Marker PDF Extraction",
    version="1.0.0",
    description="API for PDF processing using Marker",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.post("/extract")
def extract(pdf: UploadFile = File(...)):
    # Vérification du type
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. PDF required.")

    # Création d'un répertoire de travail temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        # Sauvegarde du PDF
        input_pdf_path = os.path.join(tmpdir, pdf.filename)
        with open(input_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

        try:
            result = extract_pdf(input_pdf_path, tmpdir, artifact_dict)
        except PdfConversionError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except MarkerConversionError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)