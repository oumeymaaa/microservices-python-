"""
post_processor.py — correction intelligente OCR CIN Tunisienne
"""
import re
from ocr_parser import normalize_arabic, WILAYAS_TUNISIE, DELEGATION_TO_WILAYA


# ❌ mots NON tunisiens détectés dans ton cas réel (anti-hallucination)
FORBIDDEN_PLACES = {
    "بجاية", "alger", "algérie", "algerie", "paris", "france"
}


def fix_cin_number(cin: str) -> str:
    """Force CIN valide (7 ou 8 chiffres uniquement)."""
    if not cin:
        return ""

    digits = re.sub(r"\D", "", cin)
    if len(digits) in [7, 8]:
        return digits
    return ""


def fix_date(date_str: str) -> str:
    """Sécurise la date."""
    if not date_str:
        return ""

    date_str = date_str.strip()
    if len(date_str) < 6:
        return ""

    return date_str


def is_forbidden_place(place: str) -> bool:
    """Détecte lieux hallucination OCR."""
    if not place:
        return False

    norm = normalize_arabic(place.lower())
    return any(f in norm for f in FORBIDDEN_PLACES)


def fix_place(place: str) -> str:
    """Nettoyage fort du lieu de naissance."""
    if not place:
        return ""

    place = place.strip()

    # suppression hallucinations
    if is_forbidden_place(place):
        return ""

    # mapping direct gouvernorat
    norm = normalize_arabic(place)

    for w in WILAYAS_TUNISIE:
        if normalize_arabic(w) in norm:
            return w

    for d, w in DELEGATION_TO_WILAYA.items():
        if normalize_arabic(d) in norm:
            return w

    return place


def fix_name(name: str) -> str:
    """Nettoyage nom/prénom OCR."""
    if not name:
        return ""

    name = normalize_arabic(name)

    # supprimer chiffres
    name = re.sub(r"\d+", "", name)

    # nettoyer espaces multiples
    name = " ".join(name.split())

    # trop court = invalide
    if len(name) < 2:
        return ""

    return name


def post_process_ocr_result(parsed: dict) -> dict:
    """
    Correction finale après OCR parsing
    """
    data = parsed["extracted_data"]

    data["id_number"] = fix_cin_number(data.get("id_number", ""))
    data["last_name"] = fix_name(data.get("last_name", ""))
    data["first_name"] = fix_name(data.get("first_name", ""))
    data["date_of_birth"] = fix_date(data.get("date_of_birth", ""))
    data["place_of_birth"] = fix_place(data.get("place_of_birth", ""))

    return parsed