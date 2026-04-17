import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
import json
import sys
sys.path.insert(0, 'C:\\Users\\user\\Desktop\\qari_ocr_test')

from process_document import process_document, run_ocr, get_model
from image_processing import preprocess_image
from tempfile import NamedTemporaryFile
from ocr_parser import parse_ocr_output

from config import image_config

stack = get_model()
img = preprocess_image("test.jpg")

with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    img.save(tmp.name, quality=image_config.jpeg_quality)
    temp_path = tmp.name

raw = run_ocr(stack, temp_path)
print("=== RAW ===")
if "assistant" in raw:
    m = re.search(r'assistant[\s\n\r]*(\{[\s\S]*?\})', raw)
    if m:
        print(m.group(1))
print("=========")