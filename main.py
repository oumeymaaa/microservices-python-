"""
main.py — CLI pour le traitement OCR CIN Tunisienne
"""
import sys
import logging
from pathlib import Path
from typing import Optional

from config import app_config, image_config, model_config
from process_document import process_document, save_json

logging.basicConfig(
    level=getattr(logging, app_config.log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_IMAGE = "cin.jpg"
DEFAULT_OUTPUT = "result.json"


def parse_args() -> tuple[str, str]:
    """Parse les arguments de la ligne de commande."""
    if len(sys.argv) < 2:
        return DEFAULT_IMAGE, DEFAULT_OUTPUT

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    return image_path, output_path


def main():
    image_path, output_path = parse_args()

    if not Path(image_path).exists():
        logger.error(f"Image non trouvée: {image_path}")
        print(f"Erreur: Image non trouvée: {image_path}")
        sys.exit(1)

    logger.info(f"Traitement de: {image_path}")

    try:
        result = process_document(image_path)
        save_json(output_path, result)

        logger.info(f"Résultat sauvegardé: {output_path}")

        if result.get("success"):
            print(f"✓ Extraction réussie (confiance: {result['structured_data']['confidence_score']:.0%})")
            data = result["extracted_data"]
            print(f"  CIN: {data['id_number']}")
            print(f"  Nom: {data['last_name']}")
            print(f"  Prénom: {data['first_name']}")
            print(f"  Naissance: {data['date_of_birth']}")
            print(f"  Lieu: {data['place_of_birth']}")
        else:
            print(f"✗ Extraction incomplète (confiance: {result['structured_data']['confidence_score']:.0%})")
            if result["structured_data"].get("validation_errors"):
                print("  Erreurs:")
                for err in result["structured_data"]["validation_errors"]:
                    print(f"    - {err}")

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        print(f"Erreur lors du traitement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
