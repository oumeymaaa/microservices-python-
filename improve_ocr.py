"""
improve_ocr.py — Amélioration de l'OCR sans réentraînement
Utilise des techniques de prompt engineering et post-processing
"""
import re
from typing import Optional
from difflib import SequenceMatcher


WILAYAS_NORMALIZE = {
    # Anglais → Arabe
    "tunis": "تونس", "tounes": "تونس",
    "ariana": "أريانة", "arianna": "أريانة",
    "ben arous": "بن عروس", "benarous": "بن عروس",
    "manouba": "منوبة",
    "nabeul": "نابل", "nabel": "نابل",
    "zaghouan": "زغوان", "zaghwan": "زغوان",
    "bizerte": "بنزرت", "bizert": "بنزرت",
    "beja": "باجة", "baja": "باجة", "bejah": "باجة",
    "jendouba": "جندوبة",
    "le kef": "الكاف", "kef": "الكاف", "el kef": "الكاف",
    "siliana": "سليانة",
    "kairouan": "القيروان", "kairawan": "القيروان",
    "kasserine": "القصرين",
    "sidi bouzid": "سيدي بوزيد", "sidibou zid": "سيدي بوزيد",
    "sousse": "سوسة", "susa": "سوسة",
    "monastir": "المنستير",
    "mahdia": "المهدية",
    "sfax": "صفاقس", "sfaix": "صفاقس",
    "gafsa": "قفصة",
    "tozeur": "توزر",
    "kebili": "قبلي",
    "gabes": "قابس", "gabès": "قابس",
    "medenine": "مدنين",
    "tataouine": "تطاوين", "tatawin": "تطاوين",
}

WILAYAS_ARABIC = {
    "تونس", "أريانة", "بن عروس", "منوبة", "نابل", "زغوان", "بنزرت",
    "باجة", "جندوبة", "الكاف", "سليانة", "القيروان", "القصرين",
    "سيدي بوزيد", "سوسة", "المنستير", "المهدية", "صفاقس",
    "قفصة", "توزر", "قبلي", "قابس", "مدنين", "تطاوين"
}

# Erreurs OCR courantes sur les chiffres
CIN_DIGIT_CORRECTIONS = {
    "O": "0", "o": "0", "I": "1", "l": "1",
    "Z": "2", "S": "5", "B": "8", "G": "6",
}


def normalize_arabic(text: str) -> str:
    """Normalise les caractères arabes."""
    replacements = {
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و", "ئ": "ي",
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.strip()


def similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Vérifie si deux chaînes sont similaires."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def correct_cin_digits(raw: str) -> str:
    """
    Corrige les confusions OCR fréquentes dans le numéro CIN
    (ex: O→0, l→1, Z→2).
    """
    corrected = ""
    for ch in raw:
        corrected += CIN_DIGIT_CORRECTIONS.get(ch, ch)
    # Garder seulement les chiffres
    digits = re.sub(r'\D', '', corrected)
    if re.match(r'^\d{7,8}$', digits):
        return digits
    return raw


def correct_wilaya(place: str) -> str:
    """
    Corrige le lieu de naissance en utilisant la similarité.
    Retourne le gouvernorat arabe canonique si trouvé, sinon le lieu original.
    """
    if not place:
        return place

    normalized = normalize_arabic(place.strip())

    # 1. Vérification directe
    if normalized in {normalize_arabic(w) for w in WILAYAS_ARABIC}:
        return place

    # 2. Correspondance par similarité (seuil 0.7)
    for eng, arabic in WILAYAS_NORMALIZE.items():
        if (similar(normalized, eng, 0.7)
                or similar(normalized, normalize_arabic(arabic), 0.7)
                or eng.lower() in normalized.lower()
                or normalized.lower() in eng.lower()):
            return arabic

    # 3. Correspondance partielle pour les lieux composés
    for arabic in WILAYAS_ARABIC:
        if normalize_arabic(arabic) in normalized:
            return arabic

    return place


def correct_arabic_name(text: str) -> str:
    """
    Nettoie un nom arabe des artefacts OCR courants :
    - supprime les chiffres isolés
    - supprime la ponctuation parasite
    - normalise les espaces
    """
    if not text:
        return text
    # Supprimer les chiffres qui ne devraient pas être dans un nom
    cleaned = re.sub(r'\d+', '', text)
    # Supprimer la ponctuation sauf le tiret (noms composés)
    cleaned = re.sub(r'[^\u0600-\u06FF\s\-]', '', cleaned)
    # Normaliser les espaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def post_process_ocr_result(raw_data: dict) -> dict:
    """
    Applique les corrections post-OCR sur les données extraites.
    - Corrige le numéro CIN (confusions chiffres/lettres)
    - Corrige et normalise le lieu de naissance vers le gouvernorat
    - Nettoie les noms arabes
    """
    if "extracted_data" not in raw_data:
        return raw_data

    data = raw_data["extracted_data"]

    # Correction CIN
    if data.get("id_number"):
        data["id_number"] = correct_cin_digits(data["id_number"])

    # Correction lieu de naissance
    if data.get("place_of_birth"):
        data["place_of_birth"] = correct_wilaya(data["place_of_birth"])

    # Nettoyage noms arabes
    if data.get("last_name"):
        data["last_name"] = correct_arabic_name(data["last_name"])
    if data.get("first_name"):
        data["first_name"] = correct_arabic_name(data["first_name"])

    raw_data["extracted_data"] = data
    return raw_data


# ── Prompt OCR amélioré ────────────────────────────────────────────────────
# Ce prompt impose un format strict 5-lignes et liste les gouvernorats valides,
# ce qui contraint le modèle et simplifie le parsing.
IMPROVED_PROMPT = """
Tu es un système OCR spécialisé pour la Carte d'Identité Nationale tunisienne (CIN).

⚠️ IMPORTANT :
- Répond UNIQUEMENT en JSON valide
- PAS de texte, PAS de markdown, PAS de ```json
- PAS d'explication
- PAS de phrases
- PAS de liste hors JSON

📌 FORMAT OBLIGATOIRE :

{
  "cin": "",
  "last_name": "",
  "first_name": "",
  "date_of_birth": "",
  "place_of_birth": ""
}

📌 RÈGLES :
- CIN = 7 ou 8 chiffres uniquement
- Nom = اللقب (ou dernier nom arabe)
- Prénom = الاسم
- Date = format libre (sera normalisé après)
- Lieu = gouvernorat tunisien si possible
- Si une info manque → ""

⚠️ NE JAMAIS AJOUTER AUTRE CHOSE QUE LE JSON
"""


if __name__ == "__main__":
    test_cases = [
        {
            "extracted_data": {
                "id_number": "O2225745",
                "last_name": "بن كرم3",
                "first_name": "عبد الرؤوف",
                "date_of_birth": "24 octobre 1957",
                "place_of_birth": "تبرسق بجدة"
            }
        },
        {
            "extracted_data": {
                "id_number": "12345678",
                "last_name": "الزواوي",
                "first_name": "محمد",
                "date_of_birth": "01 janvier 1985",
                "place_of_birth": "sfax"
            }
        },
    ]

    print("=== Test post-processing ===\n")
    for i, test in enumerate(test_cases, 1):
        print(f"--- Cas {i} ---")
        print(f"Avant : {test['extracted_data']}")
        corrected = post_process_ocr_result(test)
        print(f"Après : {corrected['extracted_data']}\n")