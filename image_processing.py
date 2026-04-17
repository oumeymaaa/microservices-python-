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


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _four_point_transform(image_array: np.ndarray, points: np.ndarray) -> np.ndarray:
    import cv2

    rect = _order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = int(max(width_top, width_bottom))

    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)
    max_height = int(max(height_right, height_left))

    if max_width <= 0 or max_height <= 0:
        return image_array

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image_array, matrix, (max_width, max_height))


def _box_points_from_rect(rect: tuple) -> np.ndarray:
    import cv2

    box = cv2.boxPoints(rect)
    return np.array(box, dtype="float32")


def crop_document(img: Image.Image) -> Image.Image:
    """
    Tente de recadrer automatiquement la carte d'identitÃ©.
    Si aucun contour plausible n'est trouvÃ©, retourne l'image originale.
    """
    try:
        import cv2

        image_array = np.array(img)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = img.width * img.height
        best_quad = None
        best_area = 0.0

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue

            contour_points = approx.reshape(4, 2).astype("float32")
            x, y, w, h = cv2.boundingRect(approx)
            area = float(w * h)
            if area < image_area * 0.08:
                continue

            aspect_ratio = w / max(h, 1)
            if not 0.55 <= aspect_ratio <= 2.2:
                continue

            if area > best_area:
                best_quad = contour_points
                best_area = area

        if not best_quad:
            for contour in contours:
                contour_area = float(cv2.contourArea(contour))
                if contour_area < image_area * 0.01 or contour_area > image_area * 0.30:
                    continue

                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), _ = rect
                if width <= 0 or height <= 0:
                    continue

                long_side = max(width, height)
                short_side = min(width, height)
                aspect_ratio = long_side / max(short_side, 1.0)
                box_area = float(width * height)
                fill_ratio = contour_area / max(box_area, 1.0)

                if not 1.2 <= aspect_ratio <= 2.2:
                    continue
                if fill_ratio < 0.45:
                    continue
                if box_area > best_area:
                    best_quad = _box_points_from_rect(rect)
                    best_area = box_area

        if not best_quad:
            return img

        warped = _four_point_transform(image_array, best_quad)
        if warped.size == 0:
            return img

        warped_image = Image.fromarray(warped)
        if warped_image.height > warped_image.width:
            warped_image = warped_image.rotate(90, expand=True)

        return warped_image.resize((960, 600), Image.LANCZOS)
    except ImportError:
        return img
    except Exception:
        return img


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

    img = crop_document(img)
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
    thumbnail = img.copy()
    thumbnail.thumbnail((max_size, max_size), Image.LANCZOS)
    return thumbnail
