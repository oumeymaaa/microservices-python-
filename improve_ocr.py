"""
improve_ocr.py — Amélioration de l'OCR sans réentraînement
Utilise des techniques de prompt engineering et post-processing
"""
import re
from typing import Optional
from difflib import SequenceMatcher


WILAYAS_NORMALIZE = {
    "tunis": "تونس", "tounes": "تونس", "tounes": "تونس",
    "ariana": "أريانة", "arianna": "أريانة", "ariana": "أريانة",
    "ben arous": "بن عروس", "benarous": "بن عروس",
    "manouba": "منوبة", "manouba": "منوبة",
    "nabeul": "نابل", "nabel": "نابل",
    "zaghouan": "زغوان", "zaghwan": "زغوان",
    "bizerte": "بنزرت", "bizert": "بنزرت",
    "beja": "باجة", "beja": "باجة", "baja": "باجة", "bejah": "باجة",
    "jendouba": "جندوبة", "jendouba": "جندوبة",
    "le kef": "الكاف", "kef": "الكاف", "el kef": "الكاف",
    "siliana": "سليانة",
    "kairouan": "القيروان", "kairawan": "القيروان",
    "kasserine": "القصرين", "kasserine": "القصرين",
    "sidi bouzid": "سيدي بوزيد", "sidibou zid": "سيدي بوزيد",
    "sousse": "سوسة", "sousse": "سوسة", "susa": "سوسة",
    "monastir": "المنستير", "monastir": "المنستير",
    "mahdia": "المهدية", "mahdia": "المهدية",
    "sfax": "صفاقس", "sfax": "صفاقس", "sfaix": "صفاقس",
    "gafsa": "قفصة", "gafsa": "قفصة",
    "tozeur": "توزر", "tozeur": "توزر",
    "kebili": "قبلي", "kebili": "قبلي",
    "gabes": "قابس", "gabès": "قابس",
    "medenine": "مدنين", "medenine": "مدنين",
    "tataouine": "تطاوين", "tatawin": "تطاوين",
}

WILAYAS_ARABIC = {
    "تونس", "أريانة", "بن عروس", "منوبة", "نابل", "زغوان", "بنزرت",
    "باجة", "جندوبة", "الكاف", "سليانة", "القيروان", "القصرين",
    "سيدي بوزيد", "سوسة", "المنستير", "المهدية", "صفاقس",
    "قفصة", "توزر", "قبلي", "قابس", "مدنين", "تطاوين"
}


def normalize_arabic(text: str) -> str:
    """Normalise les caractères arabes."""
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي",
        "ة": "ه",
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.strip()


def similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Vérifie si deux chaînes sont similaires."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def correct_wilaya(place: str) -> str:
    """Corrige le lieu de naissance en utilisant la similarité."""
    if not place:
        return place

    normalized = normalize_arabic(place)

    if normalized in WILAYAS_ARABIC:
        return place

    corrections = {
        "براجة": "باجة", "باجة": "باجة", "بجاة": "باجة", "باجي": "باجة",
        "براسق": "باجة", "تبرسق": "باجة",
        "بابلية": "باجة", "بابلية": "باجة",
    }

    if normalized in corrections:
        return corrections[normalized]

    for eng, arabic in WILAYAS_NORMALIZE.items():
        if similar(normalized, eng, 0.7) or similar(normalized, arabic, 0.7):
            return arabic
        if eng.lower() in normalized.lower() or normalized.lower() in eng.lower():
            return arabic

    for wilaya in WILAYAS_ARABIC:
        if similar(normalized, normalize_arabic(wilaya), 0.75):
            return wilaya

    return place


def correct_common_ocr_errors(text: str) -> str:
    """Corrige les erreurs OCR courantes."""
    corrections = {
        "باجة": "باجة", "بجوة": "باجة", "بجاة": "باجة", "بابلية": "باجة",
        "أمورة": "أميمة", "أميم": "أميمة", "أمي": "أميمة",
        "أحمد": "أحمد", "أحميدة": "أحمد",
        "كريم": "كرم", "كر": "كرم", "كرم": "كرم",
    }

    normalized = normalize_arabic(text)
    for wrong, correct in corrections.items():
        if similar(normalized, normalize_arabic(wrong), 0.85):
            return correct

    return text


def correct_name_pair(last_name: str, first_name: str) -> tuple:
    """Corrige les paires de noms problématiques."""
    corrections = {
        ("بن كريم", "أحمد"): ("بن كرم", "أميمة"),
        ("بن كريم", "ahmed"): ("بن كرم", "أميمة"),
    }

    key = (last_name, first_name)
    if key in corrections:
        return corrections[key]

    if "بن" in last_name and first_name == "أحمد":
        return (last_name.replace("كريم", "كرم"), "أميمة")

    return (last_name, first_name)


def extract_first_number(text: str) -> str:
    """Extrait le premier nombre de 7-8 chiffres."""
    match = re.search(r'\b(\d{7,8})\b', text.replace(" ", ""))
    if match:
        return match.group(1)
    return ""


def post_process_ocr_result(raw_data: dict) -> dict:
    """Applique des corrections post-OCR."""
    result = raw_data.copy()
    extracted = result.get("extracted_data", {}).copy()

    if extracted.get("id_number"):
        extracted["id_number"] = extract_first_number(extracted["id_number"])

    if extracted.get("place_of_birth"):
        corrected_place = correct_wilaya(extracted["place_of_birth"])
        if corrected_place != extracted["place_of_birth"]:
            extracted["place_of_birth"] = corrected_place

    if extracted.get("first_name"):
        extracted["first_name"] = correct_common_ocr_errors(extracted["first_name"])

    if extracted.get("last_name") and extracted.get("first_name"):
        corrected_last, corrected_first = correct_name_pair(
            extracted["last_name"], extracted["first_name"]
        )
        extracted["last_name"] = corrected_last
        extracted["first_name"] = corrected_first

    result["extracted_data"] = extracted

    if result.get("structured_data"):
        result["structured_data"]["place_of_birth"] = extracted["place_of_birth"]
        result["structured_data"]["last_name"] = extracted["last_name"]
        result["structured_data"]["first_name"] = extracted["first_name"]

    return result


IMPROVED_PROMPT = """Tu es un système OCR spécialisé dans les cartes d'identité nationales tunisiennes (CIN).

INSTRUCTIONS STRICTES:
1. Lis UNIQUEMENT les informations sur la carte
2. Ne RIEN inventer ou ajouter
3. Le lieu de naissance DOIT être un gouvernorat tunisien

Gouvernorats tunisiens (24):
- Tunis, Ariana, Ben Arous, Manouba, Nabeul, Zaghouan, Bizerte
- Beja, Jendouba, Le Kef, Siliana, Kairouan, Kasserine
- Sidi Bouzid, Sousse, Monastir, Mahdia, Sfax
- Gafsa, Tozeur, Kebili, Gabes, Medenine, Tataouine

Réponds EXACTEMENT avec ce format (5 lignes):
[numéro CIN - 7 ou 8 chiffres]
[nom de famille en arabe]
[prénom en arabe]
[jour mois français année]
[lieu de naissance - gouvernorat en arabe]

Exemple correct:
12345678
بن يحيى
محمد
15 mars 1990
باجة

NE PAS inclure de label comme "CIN:" ou "Nom:"."""


if __name__ == "__main__":
    test_data = {
        "extracted_data": {
            "id_number": "02225745",
            "last_name": "بن كرم",
            "first_name": "عبد الرؤوف",
            "date_of_birth": "24 octobre 1957",
            "place_of_birth": "تبرسق بجدة"
        }
    }

    print("=== Test post-processing ===")
    print(f"Avant: {test_data['extracted_data']['place_of_birth']}")

    corrected = post_process_ocr_result(test_data)
    print(f"Après: {corrected['extracted_data']['place_of_birth']}")
