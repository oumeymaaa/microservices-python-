import sys
import logging
from pathlib import Path

from config import app_config
from process_document import process_document
import json

logging.basicConfig(
    level=getattr(logging, app_config.log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

result = process_document("test.jpg")

print("=== RESULT ===")
print("Success:", result.get("success"))
print("Confidence:", result.get("structured_data", {}).get("confidence_score"))
print("Extracted data:", result.get("extracted_data"))
print("Raw text preview:", result.get("raw_text", "")[:100])
print("=================")