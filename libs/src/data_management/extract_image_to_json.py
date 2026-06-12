import json
import os
import time

import torch

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

    # Détection du device : GPU si CUDA disponible, sinon CPU
    use_gpu = torch.cuda.is_available()
    print(f"Device marker : {'cuda (' + torch.cuda.get_device_name(0) + ')' if use_gpu else 'cpu'}")

    # Batch sizes selon le device.
    # NB : le LLM (use_llm) est déporté sur le marker_proxy (PROXY_URL), donc le GPU
    # local ne sert qu'aux modèles surya -> on a de la marge VRAM sur les 16 Go d'une T4.
    if use_gpu:
        # Réglages pour une T4 16 Go (modèles surya uniquement en local).
        # Calibré au-dessus des défauts CUDA de surya (recognition 256 / detection 32)
        # car le LLM est déporté sur le proxy : la VRAM est quasi entièrement libre.
        # À ajuster en surveillant `nvidia-smi` (cible ~11-12 Go / 15 Go).
        batch_sizes = {
            "detection_batch_size": 36,
            "recognition_batch_size": 224,
            "layout_batch_size": 24,
            "table_rec_batch_size": 28,
            "equation_batch_size": 16,
        }
    else:
        # Réglages conservateurs pour CPU
        batch_sizes = {
            "detection_batch_size": 6,
            "recognition_batch_size": 64,
            "layout_batch_size": 6,
            "table_rec_batch_size": 6,
            "equation_batch_size": 6,
        }

    # Configuration Marker
    config = {
        "output_format": "json",
        # force_ocr=False : marker décide page par page. Les pages scan/image (sans
        # couche texte exploitable) ou à couche texte jugée mauvaise sont quand même
        # OCR-isées (cf. LineBuilder.get_all_lines) ; seules les pages de vrai texte
        # numérique sont lues directement -> gros gain sur des PDFs hybrides.
        "force_ocr": False,
        "use_llm": True,
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_base_url": os.getenv("PROXY_URL"),
        "openai_model": "gemma4-26b-moe",
        "openai_api_key": os.getenv("REAL_LLM_API_KEY"),
        "timeout": 99999,
        # --- parallélisme intra-document ---
        "pdftext_workers": 8,
        # --- batch sizes (adaptés au device détecté) ---
        **batch_sizes,
        # Parallélisme des appels LLM (réseau, via le marker_proxy) : indépendant du GPU.
        # À calibrer selon ce que le LLM distant tolère (sinon 429 / timeouts).
        "max_concurrency": 15,
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