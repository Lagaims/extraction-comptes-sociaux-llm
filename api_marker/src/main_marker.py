from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

from data_management.pdf_to_image import pdf_to_image


app = FastAPI(
    title="API Marker PDF Extraction",
    version="1.0.0",
    description="API for PDF processing using Marker",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
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
        
        # Conversion du PDF en image
        try:
            image_path = pdf_to_image(input_pdf_path, tmpdir)
            print(f"PDF converti en image: {image_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur de conversion PDF vers image: {str(e)}")
        
        # Configuration de Marker pour produire du JSON et forcer l'OCR + LLM
        config = {
            "output_format": "json",
            "force_ocr": True,
            "use_llm": True,
            "llm_service": "marker.services.openai.OpenAIService",
            "openai_base_url": os.getenv("PROXY_URL"),
            "openai_model": "gemma3:27b",
            "openai_api_key": os.getenv("REAL_LLM_API_KEY"),
            "timeout": 99999,
        }
        
        parser = ConfigParser(config)
        
        # Instanciation du converter
        converter = PdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=parser.get_processors(),
            renderer=parser.get_renderer(),
            llm_service=parser.get_llm_service()
        )
        
        # Exécution de la conversion (on continue à utiliser le PDF original pour Marker)
        try:
            rendered = converter(input_pdf_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Marker conversion failed: {e}")
        
        # Le rendu JSON complet
        result = rendered.model_dump()
        
        # Ajout des informations sur l'image générée dans la réponse
        result["image_info"] = {
            "image_generated": True,
            "image_filename": os.path.basename(image_path),
            "image_size_bytes": os.path.getsize(image_path)
        }
        
        return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)