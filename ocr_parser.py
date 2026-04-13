"""
ocr_parser.py — Parsing et validation OCR pour CIN Tunisienne
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from config import validation_config


IGNORE_PATTERNS = re.compile(
    r"الجمهورية|بطاقة|التعريف|التعرف|national|republic|tunisie|tunisia|"
    r"carte|identite|identity|république",
    re.IGNORECASE,
)

LABEL_ONLY = {
    "الاسم", "اللقب", "اسم", "لقب", "الاسم:", "اللقب:",
    "تاريخ الولادة", "مكان الولادة", "تاريخ", "مكان",
    "الجنس", "الرقم", "الرقم الوطني", "رقم البطاقة",
}

MONTH_MAP: dict[str, str] = {
    "جانفي": "01", "يناير": "01",
    "فيفري": "02", "فبراير": "02",
    "مارس": "03",
    "أفريل": "04", "افريل": "04", "ابريل": "04", "نيسان": "04",
    "ماي": "05", "مايو": "05",
    "جوان": "06", "يونيو": "06",
    "جويلية": "07", "يوليو": "07",
    "أوت": "08", "اوت": "08", "أغسطس": "08",
    "سبتمبر": "09",
    "أكتوبر": "10", "اكتوبر": "10",
    "نوفمبر": "11",
    "ديسمبر": "12", "ديسمبار": "12",
}

MONTH_NAME_MAP: dict[str, str] = {
    "01": "janvier", "02": "février", "03": "mars",
    "04": "avril", "05": "mai", "06": "juin",
    "07": "juillet", "08": "août", "09": "septembre",
    "10": "octobre", "11": "novembre", "12": "décembre",
}

WILAYAS_TUNISIE: set[str] = {
    "تونس", "أريانة", "بن عروس", "منوبة", "نابل", "زغوان", "بنزرت",
    "باجة", "جندوبة", "الكاف", "سليانة", "القيروان", "القصرين",
    "سيدي بوزيد", "سوسة", "المنستير", "المهدية", "صفاقس",
    "قفصة", "توزر", "قبلي", "قابس", "مدنين", "تطاوين",
    "tunis", "ariana", "ben arous", "manouba", "nabeul", "zaghouan", "bizerte",
    "beja", "jendouba", "le kef", "kef", "siliana", "kairouan", "kasserine",
    "sidi bouzid", "sousse", "monastir", "mahdia", "sfax",
    "gafsa", "tozeur", "kebili", "gabes", "medenine", "tataouine",
}

CIN_PATTERN = re.compile(r"^\d{7,8}$")

DATE_PATTERNS = [
    re.compile(r"(\d{1,2})\s+([\u0600-\u06FFa-zA-Z]+)\s+(\d{4})"),
    re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})"),
    re.compile(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})"),
]


@dataclass
class CinExtractedData:
    id_number: str = ""
    last_name: str = ""
    first_name: str = ""
    date_of_birth: str = ""
    place_of_birth: str = ""
    raw_date: str = ""
    all_lines: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    is_valid: bool = False
    confidence_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


def normalize_arabic(text: str) -> str:
    """Normalise les caractères arabes pour la comparaison."""
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


def clean_ocr_text(raw_text: str) -> list[str]:
    """Nettoie le texte OCR et retourne les lignes utiles."""
    lines = []
    label_patterns = [
        r"^\d+[\.\)]\s*", r"^الاسم\s*", r"^لقب\s*", r"^اللقب\s*", r"^اسم\s*",
        r"^الرقم\s*", r"^تاريخ\s*", r"^مكان\s*",
        r"^CIN\s*:?\s*", r"^رقم\s*",
        r"^Certainly\s*,?\s*", r"^Surely\s*,?\s*", r"^Here\s+is\s*:?\s*",
    ]

    for line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", line.strip())
        if not line:
            continue
        if IGNORE_PATTERNS.search(line):
            continue
        normalized = normalize_arabic(line)
        if normalized in {normalize_arabic(l) for l in LABEL_ONLY}:
            continue
        for pattern in label_patterns:
            line = re.sub(pattern, "", line, flags=re.IGNORECASE)
        if ":" in line:
            parts = line.split(":", 1)
            value = parts[1].strip() if len(parts) > 1 else ""
            if value:
                line = value
            else:
                continue
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def extract_cin_number(lines: list[str]) -> tuple[str, int]:
    """Extrait le numéro CIN (7-8 chiffres)."""
    for i, line in enumerate(lines):
        cleaned = re.sub(r"\s", "", line.strip())
        if CIN_PATTERN.match(cleaned):
            return cleaned, i
    return "", -1


def extract_date(line: str) -> str:
    """Extrait et formate une date de naissance."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(line)
        if match:
            groups = match.groups()
            if len(groups) == 3 and groups[0] and groups[2]:
                if len(groups[0]) == 4:
                    year, month, day = groups[0], groups[1], groups[2]
                else:
                    day, month_or_name, year = groups[0], groups[1], groups[2]
                    if month_or_name.isdigit():
                        month = month_or_name.zfill(2)
                    else:
                        month = MONTH_MAP.get(
                            normalize_arabic(month_or_name),
                            month_or_name
                        )
                day = day.zfill(2)
                return f"{day} {MONTH_NAME_MAP.get(month, month)} {year}"
    return ""


def extract_place_of_birth(
    lines: list[str],
    used_indices: set[int],
    cin_index: int
) -> tuple[str, int]:
    """Extrait le lieu de naissance (gouvernorat ou autre)."""
    for i, line in enumerate(lines):
        if i in used_indices or i <= cin_index:
            continue
        normalized = normalize_arabic(line)
        if len(line.strip()) >= 2:
            return line.strip(), i
    return "", -1


def extract_names(
    lines: list[str],
    used_indices: set[int],
    cin_index: int
) -> tuple[str, str]:
    """Extrait le nom et prénom."""
    remaining = [
        (i, l) for i, l in enumerate(lines)
        if i not in used_indices
        and i > cin_index
        and len(l) >= 2
        and not l.isdigit()
    ]

    last_name = ""
    first_name = ""

    if remaining:
        last_name = remaining[0][1]
    if len(remaining) > 1:
        first_name = remaining[1][1]
    elif remaining:
        parts = remaining[0][1].split()
        if len(parts) >= 2:
            last_name = " ".join(parts[: len(parts) // 2])
            first_name = " ".join(parts[len(parts) // 2:])

    return last_name, first_name


def parse_ocr_output(raw_text: str) -> dict:
    """Parse le texte OCR brut et extrait les données CIN."""
    lines = clean_ocr_text(raw_text)
    data = CinExtractedData(all_lines=lines)

    data_dict = {}
    remaining_lines = []

    for line in lines:
        if not line.strip():
            continue
        clean_line = line.strip()
        digits_only = re.sub(r"\D", "", clean_line)
        if re.match(r"^\d{7,8}$", digits_only):
            data_dict["id_number"] = digits_only
        elif re.search(r"\d{1,2}\s+(جانفي|فيفري|مارس|أفريل|ماي|جوان|جويلية|أوت|سبتمبر|أكتوبر|نوفمبر|ديسمبر|أفريل|ابريل|février|janvier|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)", line, re.IGNORECASE):
            data_dict["date_of_birth"] = line
        elif not data_dict.get("id_number"):
            if re.match(r"^\d", clean_line):
                digits = re.sub(r"\D", "", clean_line)
                if 7 <= len(digits) <= 8:
                    data_dict["id_number"] = digits
                else:
                    remaining_lines.append(line)
            else:
                remaining_lines.append(line)
        elif not data_dict.get("last_name"):
            data_dict["last_name"] = line
        elif not data_dict.get("first_name"):
            data_dict["first_name"] = line
        elif not data_dict.get("date_of_birth"):
            data_dict["date_of_birth"] = line
        elif not data_dict.get("place_of_birth"):
            data_dict["place_of_birth"] = line

    for line in remaining_lines:
        if not data_dict.get("last_name"):
            data_dict["last_name"] = line
        elif not data_dict.get("first_name"):
            data_dict["first_name"] = line
        elif not data_dict.get("date_of_birth"):
            data_dict["date_of_birth"] = line
        elif not data_dict.get("place_of_birth"):
            data_dict["place_of_birth"] = line

    data.id_number = data_dict.get("id_number", "")
    data.last_name = data_dict.get("last_name", "")
    data.first_name = data_dict.get("first_name", "")
    data.date_of_birth = extract_date(data_dict.get("date_of_birth", ""))
    data.place_of_birth = data_dict.get("place_of_birth", "")

    name_parts = data.last_name.split()
    arabic_prefixes = ["بن", "بنت", "أولاد", "عائلة"]
    if len(name_parts) >= 2:
        if name_parts[1] in arabic_prefixes:
            data.first_name = name_parts[0]
            data.last_name = " ".join(name_parts[1:])
        elif len(name_parts) > 2 and name_parts[0] in arabic_prefixes:
            data.last_name = " ".join(name_parts[:2])
            data.first_name = " ".join(name_parts[2:])
        elif not data.first_name:
            data.first_name = " ".join(name_parts[1:])
            data.last_name = name_parts[0]

    if data.first_name and re.match(r"^(بن|بنت)\s", data.first_name):
        data.last_name = data.last_name + " " + data.first_name.strip()
        data.first_name = ""

    return {
        "raw_text": raw_text,
        "extracted_data": {
            "id_number": data.id_number,
            "last_name": data.last_name,
            "first_name": data.first_name,
            "date_of_birth": data.date_of_birth,
            "place_of_birth": data.place_of_birth,
            "raw_date": data.raw_date,
        },
        "all_lines": lines,
    }


def validate_cin_data(parsed: dict) -> dict:
    """Valide les données extraites et calcule le score de confiance."""
    result = ValidationResult()
    extracted = parsed["extracted_data"]

    fields = {
        "id_number": extracted["id_number"],
        "last_name": extracted["last_name"],
        "first_name": extracted["first_name"],
        "date_of_birth": extracted["date_of_birth"],
        "place_of_birth": extracted["place_of_birth"],
    }

    for field_name, value in fields.items():
        if not value or len(value.strip()) < validation_config.min_name_length:
            result.validation_errors.append(f"{field_name}: vide ou trop court")

    if extracted["id_number"]:
        cin_len = len(extracted["id_number"])
        if cin_len not in [7, 8]:
            result.validation_errors.append(
                f"CIN: doit contenir 7 ou 8 chiffres (reçu: {cin_len})"
            )
        elif cin_len == 8:
            result.warnings.append("CIN 8 chiffres - vérification recommandée")

    if extracted["date_of_birth"] and extracted["raw_date"]:
        try:
            parts = extracted["raw_date"].split("/")
            if len(parts) == 3:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2010):
                    result.warnings.append("Date de naissance hors plage normale")
        except (ValueError, IndexError):
            result.warnings.append("Format de date non standard")

    if extracted["place_of_birth"]:
        normalized_place = normalize_arabic(extracted["place_of_birth"])
        if normalized_place not in {normalize_arabic(w) for w in WILAYAS_TUNISIE}:
            result.warnings.append(f"Lieu de naissance (non gouvernorat standard): {extracted['place_of_birth']}")

    filled_fields = sum(1 for v in fields.values() if v and len(v) >= 2)
    result.confidence_score = round(filled_fields / len(fields), 2)
    result.is_valid = result.confidence_score >= 0.6 and len(result.validation_errors) == 0

    return {
        "is_valid": result.is_valid,
        "confidence_score": result.confidence_score,
        "warnings": result.warnings,
        "validation_errors": result.validation_errors,
        "extracted_data": fields,
    }
