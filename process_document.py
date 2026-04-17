import json
import logging
import shutil
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import lru_cache
from typing import Any

from config import image_config, model_config
from image_processing import preprocess_image, image_to_base64
from improve_ocr import IMPROVED_PROMPT
from ocr_parser import parse_ocr_output, validate_cin_data

logger = logging.getLogger(__name__)

DEBUG_DIR = Path("debug")


# ─────────────────────────────
# MODEL LOAD
# ─────────────────────────────
@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_config.base_model_id)

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.base_model_id,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base, model_config.adapter_model_id)
    model.eval()

    return {
        "model": model,
        "processor": processor,
        "torch": torch,
        "device": device,
        "process_vision_info": process_vision_info
    }


# ─────────────────────────────
# CLEAN JSON OUTPUT
# ─────────────────────────────
def extract_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)

    if not match:
        return None

    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data[0]
        return data
    except:
        return None


# ─────────────────────────────
# OCR RUN
# ─────────────────────────────
def run_ocr(stack, image_path: str) -> str:
    processor = stack["processor"]
    model = stack["model"]
    torch = stack["torch"]
    process_vision = stack["process_vision_info"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": IMPROVED_PROMPT}
        ]
    }]

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img, vid = process_vision(messages)

    inputs = processor(
        text=[chat],
        images=img,
        videos=vid,
        return_tensors="pt"
    )

    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0
        )

    result = processor.decode(out[0], skip_special_tokens=True)

    json_data = extract_json(result)

    if json_data:
        return json.dumps(json_data, ensure_ascii=False)

    return result


# ─────────────────────────────
# PIPELINE
# ─────────────────────────────
def process_document(image_path: str) -> dict[str, Any]:
    logger.info(f"Processing: {image_path}")

    stack = get_model()
    img = preprocess_image(image_path)

    with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, quality=image_config.jpeg_quality)
        temp_path = tmp.name

    raw = run_ocr(stack, temp_path)

    parsed = parse_ocr_output(raw)
    validated = validate_cin_data(parsed)

    return {
        "success": validated["is_valid"],
        "raw_text": raw,
        "extracted_data": validated["extracted_data"],
        "structured_data": {
            **validated["extracted_data"],
            "confidence_score": validated["confidence_score"],
            "warnings": validated["warnings"],
            "validation_errors": validated["validation_errors"]
        },
        "image": Path(image_path).name,
        "processed_image_base64": image_to_base64(img)
    }