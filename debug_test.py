import sys
sys.path.insert(0, "C:\\Users\\user\\Desktop\\qari_ocr_test")

import re
import json
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ocr_parser import parse_ocr_output

raw_text = '''system
You are a helpful assistant.
user

Tu es un système OCR spécialisé pour la Carte d'Identité Nationale tunisienne (CIN).

assistant
{
  "cin": "14426787",
  "last_name": "بن كرم",
  "first_name": "أميمة",
  "date_of_birth": "20 أبريل 2002",
  "place_of_birth": "زغوان"
}'''

result = parse_ocr_output(raw_text)
print("Result keys:", result.keys())
print("extracted_data:", result.get("extracted_data"))