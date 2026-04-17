import re
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

assistant_match = re.search(r'assistant\s*[\n\r]*(\{.*\})', raw_text, re.DOTALL)
if assistant_match:
    json_str = assistant_match.group(1)
    print("FOUND:", json_str[:100])
    try:
        data = json.loads(json_str)
        print("PARSED:", data)
    except Exception as e:
        print("ERROR:", e)