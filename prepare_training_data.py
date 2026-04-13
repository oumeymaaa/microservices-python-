"""
prepare_training_data.py — Préparation des données d'entraînement pour améliorer l'OCR CIN
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class CinGroundTruth:
    image_path: str
    id_number: str
    last_name: str
    first_name: str
    date_of_birth: str
    place_of_birth: str
    notes: str = ""
    quality: str = "good"  # good, medium, poor
    verified: bool = False


def create_training_data_from_results(results_dir: str, output_path: str):
    """Compile les résultats OCR en données d'entraînement."""
    training_data = []
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("success") and data.get("structured_data", {}).get("confidence_score", 0) >= 0.8:
                extracted = data["extracted_data"]
                training_data.append({
                    "image_path": data.get("image", json_file.stem + ".jpg"),
                    "id_number": extracted.get("id_number", ""),
                    "last_name": extracted.get("last_name", ""),
                    "first_name": extracted.get("first_name", ""),
                    "date_of_birth": extracted.get("date_of_birth", ""),
                    "place_of_birth": extracted.get("place_of_birth", ""),
                })
        except Exception as e:
            print(f"Erreur lecture {json_file}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"Créé {len(training_data)} entrées d'entraînement -> {output_path}")
    return training_data


def create_sft_training_format(training_data: list, output_path: str):
    """Convertit les données au format SFT (Supervised Fine-Tuning) pour LLM."""
    sft_data = []
    for item in training_data:
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": "Lis cette carte d'identité nationale tunisienne et donne les informations suivantes:\n1. Numéro CIN\n2. Nom de famille\n3. Prénom\n4. Date de naissance\n5. Lieu de naissance"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": f"""CIN: {item['id_number']}
Nom: {item['last_name']}
Prénom: {item['first_name']}
Date de naissance: {item['date_of_birth']}
Lieu de naissance: {item['place_of_birth']}"""
                }
            ]
        }
        sft_data.append(conversation)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"Créé {len(sft_data)} entrées SFT -> {output_path}")
    return sft_data


def generate_additional_training_data(output_path: str, num_samples: int = 50):
    """Génère des exemples supplémentaires avec variations synthétiques."""
    import random

    wilayas = [
        "تونس", "أريانة", "بن عروس", "منوبة", "نابل", "زغوان", "بنزرت",
        "باجة", "جندوبة", "الكاف", "سليانة", "القيروان", "القصرين",
        "سيدي بوزيد", "سوسة", "المنستير", "المهدية", "صفاقس",
        "قفصة", "توزر", "قبلي", "قابس", "مدنين", "تطاوين"
    ]

    noms = ["بني يحيى", "الزواوي", "الحاج يحيى", "الشابي", "المزغني", "الهمامي", "بوحجر", "الصامتي"]
    prenoms = ["محمد", "عمر", "أحمد", "عبد الله", "ياسين", "مراد", "سامي", "كريم"]
    mois = ["جانفي", "فيفري", "مارس", "أفريل", "ماي", "جوان", "جويلية", "أوت", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]

    samples = []
    for i in range(num_samples):
        cin = f"{random.randint(1000000, 99999999)}"
        nom = random.choice(noms)
        prenom = random.choice(prenoms)
        jour = random.randint(1, 28)
        mois_choisi = random.choice(mois)
        annee = random.randint(1950, 2010)
        lieu = random.choice(wilayas)

        samples.append({
            "id_number": cin,
            "last_name": nom,
            "first_name": prenom,
            "date_of_birth": f"{jour} {mois_choisi} {annee}",
            "place_of_birth": lieu,
            "image_path": f"synthetic_cin_{i+1:03d}.jpg"
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Généré {num_samples} exemples synthétiques -> {output_path}")
    return samples


def create_validation_dataset(real_data: list, synthetic_data: list, 
                             train_ratio: float = 0.8) -> tuple:
    """Sépare les données en training et validation."""
    all_data = real_data + synthetic_data
    random.shuffle(all_data)

    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    return train_data, val_data


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Prépare les données d'entraînement OCR")
    parser.add_argument("--mode", choices=["compile", "sft", "generate", "full"],
                        default="full", help="Mode d'exécution")
    parser.add_argument("--results-dir", default="results", help="Dossier des résultats OCR")
    parser.add_argument("--output-dir", default="training_data", help="Dossier de sortie")
    parser.add_argument("--num-synthetic", type=int, default=100, help="Nb exemples synthétiques")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    if args.mode in ["compile", "full"]:
        print("=== Compilation des données réelles ===")
        real_data = create_training_data_from_results(args.results_dir, 
                                                      f"{args.output_dir}/real_data.json")

    if args.mode in ["generate", "full"]:
        print("\n=== Génération des données synthétiques ===")
        synthetic_data = generate_additional_training_data(
            f"{args.output_dir}/synthetic_data.json",
            args.num_synthetic
        )

    if args.mode == "sft":
        print("=== Conversion au format SFT ===")
        real_data = create_training_data_from_results(args.results_dir, 
                                                      f"{args.output_dir}/temp_real.json")
        create_sft_training_format(real_data, f"{args.output_dir}/sft_training.json")

    print("\n=== Données prêtes pour l'entraînement ===")
    print(f"Dossier: {args.output_dir}/")
