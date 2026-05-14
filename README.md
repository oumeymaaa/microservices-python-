# Backend Python — API KYC

Pipeline eKYC complet : OCR (EasyOCR) + Extraction visage (MediaPipe) + Comparaison faciale (InsightFace).

## Fichiers

- `api_server.py` — Serveur FastAPI (port 8000)
- `easyocr_processor.py` — OCR CIN tunisienne (EasyOCR GPU)
- `photo_extractor.py` — Extraction visage depuis la CIN (MediaPipe/OpenCV)
- `face_comparison.py` — Qualité selfie + comparaison InsightFace
- `static/faces/` — Visages extraits (servis statiquement)
- `temp_docs/` — Documents uploadés (temporaires)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```powershell
.venv\Scripts\python.exe api_server.py
```

## Endpoints

### KYC Session
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/kyc/sessions` | Créer une session |
| GET | `/kyc/sessions/{id}` | Récupérer une session |
| POST | `/kyc/sessions/{id}/document` | Uploader un document |
| POST | `/kyc/sessions/{id}/document/{docId}/extract-face` | Extraire le visage |
| POST | `/kyc/sessions/{id}/selfie` | Uploader un selfie |
| POST | `/kyc/sessions/{id}/compare-faces` | Comparer selfie vs document |

### Tests
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/ocr/cin` | OCR direct |
| POST | `/extract-face` | Extraction visage directe |
| POST | `/check-selfie-quality` | Vérifier qualité selfie |
| POST | `/compare-faces` | Comparaison directe (2 fichiers) |

## Tech

- **EasyOCR** — OCR arabe+anglais (GPU NVIDIA)
- **MediaPipe** — Détection visage CIN
- **InsightFace buffalo_l** — Embedding + comparaison faciale (ONNX GPU)
- **FastAPI** — Serveur HTTP asynchrone
