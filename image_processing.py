"""
image_processing.py — Prétraitement d'images pour OCR CIN Tunisienne
"""
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config import image_config


def load_image(path: str) -> Image.Image:
    """Charge une image depuis le chemin donné."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def auto_contrast(img: Image.Image) -> Image.Image:
    """Applique une correction automatique du contraste."""
    from PIL import ImageOps
    return ImageOps.autocontrast(img, cutoff=2)


def denoise(img: Image.Image) -> Image.Image:
    """Réduit le bruit tout en préservant les contours du texte."""
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        import cv2
        try:
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        except Exception:
            pass
    return Image.fromarray(img_array)


def sharpen_text(img: Image.Image) -> Image.Image:
    """Améliore la netteté du texte pour l'OCR."""
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(2.0)


def deskew(img: Image.Image) -> Image.Image:
    """
    Corrige automatiquement l'inclinaison de l'image.
    Utilise OpenCV pour la détection d'angle.
    """
    try:
        import cv2

        gray = np.array(img.convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(binary > 0))
        if len(coords) == 0:
            return img

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 0.3:
            return img

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            np.array(img),
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return Image.fromarray(rotated)

    except ImportError:
        return img
    except Exception:
        return img


def resize_if_needed(img: Image.Image, max_size: Optional[int] = None) -> Image.Image:
    """Redimensionne l'image si elle dépasse la taille maximale."""
    max_dim = max_size or image_config.max_size
    if max(img.size) <= max_dim:
        return img

    scale = max_dim / max(img.size)
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


def normalize_for_ocr(img: Image.Image) -> Image.Image:
    """
    Applique les adjustments pour optimiser la lecture OCR.
    """
    img = ImageEnhance.Contrast(img).enhance(image_config.contrast_factor)
    img = ImageEnhance.Sharpness(img).enhance(image_config.sharpness_factor)
    img = ImageEnhance.Brightness(img).enhance(image_config.brightness_factor)
    return img


def preprocess_image(path: str) -> Image.Image:
    """
    Pipeline complet de prétraitement pour OCR CIN.

    Étapes:
    1. Chargement
    2. Correction d'inclinaison (deskew)
    3. Redimensionnement
    4. Normalisation (contraste, netteté, luminosité)
    """
    img = load_image(path)

    if image_config.enable_deskew:
        img = deskew(img)

    img = resize_if_needed(img)
    img = normalize_for_ocr(img)

    return img


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convertit une image en chaîne base64."""
    import base64
    buffer = BytesIO()
    img.save(buffer, format=format, quality=image_config.jpeg_quality)
    return base64.b64encode(buffer.getvalue()).decode()


def create_thumbnail(img: Image.Image, max_size: int = 200) -> Image.Image:
    """Crée une miniature pour la prévisualisation."""
    return img.copy().thumbnail((max_size, max_size), Image.LANCZOS)
