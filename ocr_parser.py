"""
ocr_parser.py — Parsing et validation OCR pour CIN Tunisienne
"""
import re
import json
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
    "تاريخ الولادة", "مكان الولادة", "تاريخ", "مكان", "تاريخ:", "مكانة:", "تاريخ الزواج",
    "التاريخ الولادة", "التاريخ",
    "الجنس", "الرقم", "الرقم الوطني", "رقم البطاقة",
    "الجهازية التونسية", "بطاقة التعرف الوطنية", "المجمع العام",
    "الجهاز الوطني التونسي", "بـ طاقة التحريف الوطنية",
    "اسماء", "اسماء:", "المدينة", "المدينة:",
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
    "janvier": "01", "février": "02", "fevrier": "02", "mars": "03",
    "avril": "04", "mai": "05", "juin": "06", "juillet": "07",
    "août": "08", "aout": "08", "septembre": "09", "octobre": "10",
    "novembre": "11", "décembre": "12", "decembre": "12",
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

DELEGATION_TO_WILAYA: dict[str, str] = {
    "المرناقية": "تونس", "المنيهلة": "تونس", "حي التضامن": "تونس",
    "حلق الوادي": "تونس", "الحمامات": "نابل", "قرمبالية": "نابل",
    "سيدي ثابت": "أريانة", "رواد": "أريانة", "المنيهلة": "أريانة",
    "مدينة الطب": "بن عروس", "حمام الأنف": "بن عروس", "حمام الشط": "بن عروس",
    "بومهل": "بن عروس", "المحمدية": "بن عروس", "المروج": "بن عروس",
    "منزل بورقيبة": "بنزرت", "مطر": "بنزرت", "غار الملح": "بنزرت",
    "مجاز الباب": "باجة", "نفزة": "باجة", "عمدون": "باجة",
    "بوسالم": "جندوبة", "طبرقة": "جندوبة", "عين دراهم": "جندوبة",
    "الدهماني": "الكاف", "القصور": "الكاف", "الجريصة": "الكاف",
    "بوعرادة": "سليانة", "العروسة": "سليانة", "مكثر": "سليانة",
    "حفوز": "القيروان", "العلا": "القيروان", "الشبيكة": "القيروان",
    "سبيطلة": "القصرين", "تالة": "القصرين", "حيدرة": "القصرين",
    "جلمة": "سيدي بوزيد", "سيدي علي بن عون": "سيدي بوزيد",
    "اكودة": "سوسة", "القلعة الكبرى": "سوسة", "سيدي بو علي": "سوسة",
    "الساحلين": "المنستير", "زرمدين": "المنستير", "بني حسان": "المنستير",
    "شربان": "المهدية", "الهوارية": "المهدية", "بومرداس": "المهدية",
    "صخيرة": "صفاقس", "العامرة": "صفاقس", "جبنيانة": "صفاقس",
    "القطار": "قفصة", "المتلوي": "قفصة", "الرديف": "قفصة",
    "تمغزة": "توزر", "نفطة": "توزر", "دقاش": "توزر",
    "دوز": "قبلي", "الفوار": "قبلي", "سوق الأحد": "قبلي",
    "مارث": "قابس", "الحامة": "قابس", "متماطة": "قابس",
    "بن قردان": "مدنين", "جرجيس": "مدنين", "سيدي مخلوف": "مدنين",
    "صمار": "تطاوين", "البئر الأحمر": "تطاوين", "غمراسن": "تطاوين",
}

CIN_PATTERN = re.compile(r"^\d{7,8}$")

DATE_PATTERNS = [
    re.compile(r"(\d{1,2})\s+([\u0600-\u06FFa-zA-Zéûôàâîèùç]+)\s+(\d{4})"),
    re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})"),
    re.compile(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})"),
]

DATE_MONTHS_ALL = list(MONTH_MAP.keys())

ARABIC_PREFIXES = {"بن", "بنت", "أولاد", "عائلة", "آل", "ابن", "ابنة"}


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
    replacements = {"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.strip()


def clean_ocr_text(raw_text: str) -> list[str]:
    lines = []
    label_patterns = [
        r"^\d+[\.\)]\s*", r"^الاسم\s*", r"^لقب\s*", r"^اللقب\s*", r"^اسم\s*", r"^ال:",
        r"^الرقم\s*", r"^تاريخ\s*", r"^تاريخ:", r"^مكان\s*", r"^مكانة:", r"^CIN\s*:?\s*",
        r"^Certainly\s*,?\s*", r"^Surely\s*,?\s*", r"^Here\s+is\s*:?\s*",
        r"^العربية:", r"^المصطلح الأسماء:",
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


def normalize_extracted_value(value: str) -> str:
    normalized = value.strip()
    if normalized.upper() in {"EMPTY", "N/A", "NULL", "UNKNOWN", "?", "-", "—"}:
        return ""
    return normalized


def strip_known_label(line: str) -> str:
    patterns = [
        r"^\s*اللقب[:\s-]*", r"^\s*الاسم[:\s-]*", r"^\s*اسم[:\s-]*",
        r"^\s*لقب[:\s-]*", r"^\s*تاريخ الولادة[:\s-]*", r"^\s*مكان الولادة[:\s-]*",
    ]
    cleaned = line.strip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)
    return cleaned.strip()


def resolve_place_to_wilaya(place: str) -> str:
    if not place:
        return place
    normalized = normalize_arabic(place.strip())
    for wilaya in WILAYAS_TUNISIE:
        if normalize_arabic(wilaya) == normalized:
            return wilaya
    for delegation, wilaya in DELEGATION_TO_WILAYA.items():
        if normalize_arabic(delegation) in normalized or normalized in normalize_arabic(delegation):
            return wilaya
    for wilaya in WILAYAS_TUNISIE:
        if normalize_arabic(wilaya) in normalized and len(wilaya) >= 3:
            return wilaya
    return place


def is_date_line(line: str) -> bool:
    lower = line.lower()
    for month in DATE_MONTHS_ALL:
        if month.lower() in lower:
            return True
    if re.search(r"\d{1,2}\s+\w+\s+\d{4}", line):
        return True
    if re.search(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}", line):
        return True
    return False


def is_wilaya_line(line: str) -> bool:
    normalized = normalize_arabic(line)
    for wilaya in WILAYAS_TUNISIE:
        if normalize_arabic(wilaya) in normalized:
            return True
    for delegation in DELEGATION_TO_WILAYA:
        if normalize_arabic(delegation) in normalized:
            return True
    return False


def is_arabic_name(line: str) -> bool:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", line))
    return arabic_chars > len(line) * 0.5


def extract_cin_number(lines: list[str]) -> tuple[str, int]:
    for i, line in enumerate(lines):
        digits = re.sub(r"\D", "", line)
        if len(digits) == 7 or len(digits) == 8:
            return digits, i
    return "", -1


def extract_date(line: str) -> str:
    for pattern in DATE_PATTERNS:
        match = pattern.search(line)
        if match:
            groups = match.groups()
            if len(groups) == 3 and groups[0] and groups[2]:
                if len(groups[0]) == 4:
                    year, month_raw, day = groups[0], groups[1], groups[2]
                else:
                    day, month_raw, year = groups[0], groups[1], groups[2]
                if month_raw.isdigit():
                    month = month_raw.zfill(2)
                else:
                    month = MONTH_MAP.get(normalize_arabic(month_raw.lower()), month_raw)
                day = day.zfill(2)
                month_name = MONTH_NAME_MAP.get(month, month)
                return f"{day} {month_name} {year}"
    return ""


def parse_ocr_output(raw_text: str) -> dict:
    assistant_match = re.search(r'assistant[\s\n\r]*(\{[\s\S]*?\})', raw_text)
    if assistant_match:
        json_str = assistant_match.group(1).strip()
        special_quotes = ['\u201c', '\u201d', '\u2018', '\u2019']
        for sq in special_quotes:
            json_str = json_str.replace(sq, '"')
        try:
            data = json.loads(json_str)
            def clean_value(v):
                if isinstance(v, str):
                    v = v.strip()
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                return v

            last = clean_value(data.get("last_name", ""))
            first = clean_value(data.get("first_name", ""))

            last_norm = normalize_arabic(last)
            first_norm = normalize_arabic(first)

            has_last_prefix = "بنت" in last or "بن" in last
            has_first_prefix = "بنت" in first or "بن" in first

            # ModèleQwen lit à l'envers:
            # last = "Bent..." (devrait être first = "Amina")
            # first = "Bin..." (devrait être last = "Bin Kram")
            # → on swap si tous les deux contiennent le préfixe
            if has_last_prefix and has_first_prefix:
                last, first = first, last

            return {
                "raw_text": raw_text,
                "extracted_data": {
                    "id_number": clean_value(data.get("cin", "")),
                    "last_name": last,
                    "first_name": first,
                    "date_of_birth": clean_value(data.get("date_of_birth", "")),
                    "place_of_birth": clean_value(data.get("place_of_birth", "")),
                },
                "all_lines": [],
            }
        except Exception as e:
            pass

    lines = clean_ocr_text(raw_text)
    data = CinExtractedData(all_lines=lines)
    text_lines = [l.strip() for l in lines if l.strip() and (len(l.strip()) <= 50 or is_date_line(l.strip()) or is_wilaya_line(l.strip()))]

    id_number, cin_line_idx = extract_cin_number(text_lines)
    remaining = [(i, strip_known_label(l)) for i, l in enumerate(text_lines) if i != cin_line_idx and strip_known_label(l).strip()]

    used = set()
    raw_date, date_of_birth = "", ""
    for i, (idx, line) in enumerate(remaining):
        if is_date_line(line):
            raw_date = line
            date_of_birth = extract_date(line) or line
            used.add(i)
            break

    place_of_birth = ""
    for i, (idx, line) in enumerate(remaining):
        if i in used or not is_wilaya_line(line):
            continue
        place_of_birth = resolve_place_to_wilaya(strip_known_label(line))
        used.add(i)
        break

    if not place_of_birth:
        for i in range(len(remaining) - 1, -1, -1):
            if i not in used:
                _, line = remaining[i]
                if is_arabic_name(line) and not is_date_line(line):
                    place_of_birth = resolve_place_to_wilaya(strip_known_label(line))
                    used.add(i)
                    break

    name_lines = [(i, line) for i, (idx, line) in enumerate(remaining) if i not in used and is_arabic_name(line) and not line.isdigit()]

    first_name, last_name = "", ""
    for idx, line in name_lines:
        line_val = normalize_extracted_value(line)
        line_norm = normalize_arabic(line_val)
        has_prefix = "بنت" in line_norm or "بن" in line_norm
        if has_prefix and not last_name:
            last_name = line_val
        elif not has_prefix and not first_name:
            first_name = line_val
        elif has_prefix and last_name:
            first_name = line_val
        elif first_name:
            last_name = line_val

    data.id_number = normalize_extracted_value(id_number)
    data.last_name = normalize_extracted_value(last_name)
    data.first_name = normalize_extracted_value(first_name)
    data.raw_date = raw_date
    data.date_of_birth = date_of_birth
    data.place_of_birth = normalize_extracted_value(place_of_birth)

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
            result.validation_errors.append(f"CIN: doit contenir 7 ou 8 chiffres (reçu: {cin_len})")
        elif cin_len == 8:
            result.warnings.append("CIN 8 chiffres - vérification recommandée")

    if extracted["date_of_birth"]:
        try:
            parts = extracted["date_of_birth"].split()
            if len(parts) == 3:
                day, month_name, year = int(parts[0]), parts[1], int(parts[2])
                if not (1 <= day <= 31 and 1900 <= year <= 2010):
                    result.warnings.append("Date de naissance hors plage normale")
        except (ValueError, IndexError):
            result.warnings.append("Format de date non standard")

    if extracted["place_of_birth"]:
        normalized_place = normalize_arabic(extracted["place_of_birth"])
        known = {normalize_arabic(w) for w in WILAYAS_TUNISIE} | {normalize_arabic(d) for d in DELEGATION_TO_WILAYA}
        if normalized_place not in known:
            result.warnings.append(f"Lieu de naissance (non reconnu): {extracted['place_of_birth']}")

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