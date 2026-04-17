import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ocr_parser import parse_ocr_output, validate_cin_data
from process_document import process_document

result = process_document("test.jpg")

raw = result.get("raw_text", "")
print("RAW length:", len(raw))

if "assistant" in raw:
    print("Has assistant")

parsed = parse_ocr_output(raw)
print("Parsed data:", parsed.get("extracted_data"))

validated = validate_cin_data(parsed)
print("Validated:", validated["extracted_data"])

output = {
    "success": validated["is_valid"],
    "raw_text": raw[:200] if raw else "",
    "extracted_data": validated["extracted_data"],
    "confidence_score": validated["confidence_score"]
}

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Done")