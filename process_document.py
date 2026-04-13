"""
process_document.py — Pipeline OCR optimisé pour CIN Tunisienne
"""
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import lru_cache
from typing import Any

from config import model_config, image_config
from image_processing import preprocess_image, image_to_base64
from ocr_parser import parse_ocr_output, validate_cin_data
from improve_ocr import post_process_ocr_result

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model_stack():
    import torch
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Chargement du modèle sur: {device}")

    processor = AutoProcessor.from_pretrained(model_config.base_model_id)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.base_model_id,
        torch_dtype="auto",
        device_map="cuda:0" if device == "cuda" else "auto",
    )
    model = PeftModel.from_pretrained(base_model, model_config.adapter_model_id)
    model.eval()

    return {
        "device": device,
        "model": model,
        "processor": processor,
        "process_vision_info": process_vision_info,
        "torch": torch,
    }


def warmup_model() -> None:
    logger.info("Warming up model...")
    _get_model_stack()
    logger.info("Model warmup completed")


OCR_PROMPT = """Tu es un système OCR spécialisé dans les cartes d'identité nationales tunisiennes (CIN).

INSTRUCTIONS STRICTES:
1. Lis UNIQUEMENT les informations sur la carte
2. Ne RIEN inventer ou ajouter
3. Le lieu de naissance DOIT être un gouvernorat tunisien parmi:
   Tunis, Ariana, Ben Arous, Manouba, Nabeul, Zaghouan, Bizerte,
   Beja, Jendouba, Le Kef, Siliana, Kairouan, Kasserine,
   Sidi Bouzid, Sousse, Monastir, Mahdia, Sfax,
   Gafsa, Tozeur, Kebili, Gabes, Medenine, Tataouine

Réponds EXACTEMENT avec ce format (5 lignes, rien d'autre):
1. numéro CIN (7 ou 8 chiffres)
2. nom de famille en arabe
3. prénom en arabe
4. jour mois français année (ex: 15 mars 1990)
5. lieu de naissance en arabe

Exemple:
12345678
بن يحيى
محمد
15 mars 1990
باجة

NE PAS inclure de label comme "CIN:" ou "Nom:"."""


def _run_ocr(stack: dict, image_path: str) -> str:
    torch = stack["torch"]
    processor = stack["processor"]
    model = stack["model"]
    process_vision_info = stack["process_vision_info"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": OCR_PROMPT},
        ],
    }]

    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=img_inputs,
        videos=vid_inputs,
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=model_config.max_new_tokens,
            do_sample=False,
            temperature=model_config.temperature,
            repetition_penalty=model_config.repetition_penalty,
        )

    trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def process_document(image_path: str) -> dict[str, Any]:
    """
    Traite une image de CIN tunisienne et retourne les données extraites.

    Args:
        image_path: Chemin vers le fichier image (JPEG, PNG, etc.)

    Returns:
        dict contenant:
            - raw_text: texte brut OCR
            - extracted_data: données extraites et validées
            - structured_data: données structurées avec métadonnées
            - image: nom du fichier source
            - processed_image_base64: image prétraitée encodée
    """
    logger.info(f"Traitement de l'image: {image_path}")

    preprocessed_img = preprocess_image(image_path)

    stack = _get_model_stack()
    temp_path = None

    try:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            preprocessed_img.save(temp_path, quality=image_config.jpeg_quality)

        raw_text = _run_ocr(stack, temp_path)
        logger.debug(f"OCR brut:\n{raw_text}")

        lines = raw_text.strip().split("\n")
        if len(lines) >= 5:
            lines = [l.strip() for l in lines[:5]]
            raw_text = "\n".join(lines)

    except Exception as e:
        logger.error(f"Erreur OCR: {e}")
        raise
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    parsed_data = parse_ocr_output(raw_text)
    parsed_data = post_process_ocr_result(parsed_data)
    validated_data = validate_cin_data(parsed_data)

    result = {
        "success": validated_data["is_valid"],
        "raw_text": raw_text,
        "extracted_data": validated_data["extracted_data"],
        "structured_data": {
            **validated_data["extracted_data"],
            "confidence_score": validated_data["confidence_score"],
            "warnings": validated_data["warnings"],
            "validation_errors": validated_data["validation_errors"],
        },
        "image": Path(image_path).name,
        "processed_image_base64": image_to_base64(preprocessed_img),
    }

    logger.info(f"Traitement terminé - Validité: {result['success']}")
    return result


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
