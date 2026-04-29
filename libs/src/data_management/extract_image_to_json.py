import json
import os
import time

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

from data_management.pdf_to_image import pdf_to_image


class PdfConversionError(Exception):
    """Erreur lors de la conversion PDF -> image."""
    pass


class MarkerConversionError(Exception):
    """Erreur lors de la conversion Marker."""
    pass


def load_models() -> dict:
    """Charge les modèles Marker une seule fois au démarrage."""
    return create_model_dict()


def extract_pdf(pdf_path: str, tmpdir: str, artifact_dict: dict) -> dict:
    """Logique métier d'extraction, indépendante de FastAPI."""

    start = time.time()

    # Conversion du PDF en image
    try:
        image_path = pdf_to_image(pdf_path, tmpdir)
        print(f"PDF converti en image: {image_path}")
    except Exception as e:
        raise PdfConversionError(f"Erreur de conversion PDF vers image: {e}") from e

    # Configuration Marker
    config = {
        "output_format": "json",
        "force_ocr": True,
        "use_llm": False, # Premier test sans l'utilisation d'un LLM
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_base_url": os.getenv("PROXY_URL"),
        "openai_model": "gemma4-31b",
        "openai_api_key": os.getenv("REAL_LLM_API_KEY"),
        "timeout": 99999,
        # --- parallélisme intra-document ---
        "pdftext_workers": 8,
        # --- batch sizes (CPU, donc on reste raisonnable) ---
        "detection_batch_size": 6,
        "recognition_batch_size": 64,
        "layout_batch_size": 6,
        "table_rec_batch_size": 6,
        "equation_batch_size": 6,
        # --- concurrence LLM : LE levier principal pour ton cas ---
        "max_concurrency": 10,
    }

    parser = ConfigParser(config)
    converter = PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service(),
    )

    try:
        rendered = converter(pdf_path)
    except Exception as e:
        raise MarkerConversionError(f"Marker conversion failed: {e}") from e

    elapsed = time.time() - start
    print(f"Extraction terminée en {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # model_dump() peut échouer sur certains champs Set[...] de pydantic v2
    result = json.loads(rendered.model_dump_json())
    result["image_info"] = {
        "image_generated": True,
        "image_filename": os.path.basename(image_path),
        "image_size_bytes": os.path.getsize(image_path),
    }
    return result