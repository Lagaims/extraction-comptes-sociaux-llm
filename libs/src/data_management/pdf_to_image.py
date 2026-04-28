import pymupdf as fitz
from PIL import Image
import io
import os


def pdf_to_image(pdf_path, output_dir, dpi=300):
    """
    Convertit un PDF monopage en image
    
    Args:
        pdf_path (str): Chemin vers le fichier PDF
        output_dir (str): Répertoire de sortie pour l'image
        dpi (int): Résolution de l'image (défaut: 300 DPI)
    
    Returns:
        str: Chemin vers l'image générée
    """
    try:
        # Ouvrir le PDF
        pdf_document = fitz.open(pdf_path)
        
        # Vérifier que c'est bien un PDF monopage
        if len(pdf_document) != 1:
            raise ValueError(f"Le PDF contient {len(pdf_document)} pages. Seuls les PDFs monopages sont supportés.")
        
        # Récupérer la première (et unique) page
        page = pdf_document[0]
        
        # Définir la matrice de transformation pour la résolution
        mat = fitz.Matrix(dpi/72, dpi/72)
        
        # Convertir la page en image
        pix = page.get_pixmap(matrix=mat)
        
        # Convertir en PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Sauvegarder l'image
        image_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".png"
        image_path = os.path.join(output_dir, image_filename)
        img.save(image_path, "PNG", optimize=True)
        
        # Fermer le document PDF
        pdf_document.close()
        
        return image_path
        
    except Exception as e:
        raise Exception(f"Erreur lors de la conversion PDF vers image: {str(e)}")