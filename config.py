"""
config.py — Configuration centralisée pour l'OCR CIN Tunisienne
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


BASE_DIR = Path(__file__).parent


@dataclass
class ModelConfig:
    adapter_model_id: str = "NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct"
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.0
    repetition_penalty: float = 1.2


@dataclass
class ImageConfig:
    max_size: int = 1024
    contrast_factor: float = 1.8
    sharpness_factor: float = 2.2
    brightness_factor: float = 1.05
    jpeg_quality: int = 92
    enable_deskew: bool = True


@dataclass
class ValidationConfig:
    cin_length: tuple[int, int] = (7, 8)
    min_name_length: int = 2
    date_format: Literal["dmy", "ymd", "auto"] = "auto"


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"


model_config = ModelConfig()
image_config = ImageConfig()
validation_config = ValidationConfig()
app_config = AppConfig()