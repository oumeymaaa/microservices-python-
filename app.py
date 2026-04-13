"""
app.py — API FastAPI pour l'OCR CIN Tunisienne
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import app_config
from process_document import process_document, warmup_model

logging.basicConfig(
    level=getattr(logging, app_config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage du service OCR CIN...")
    try:
        warmup_model()
        logger.info("Modèle OCR prêt")
    except Exception as e:
        logger.warning(f"Warmup model: {e}")
    yield
    logger.info("Arrêt du service OCR CIN...")


app = FastAPI(
    title="CIN OCR API",
    description="API OCR pour Carte d'Identité Nationale Tunisienne",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "service": "cin-ocr-api",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.post("/ocr/cin")
async def scan_cin(file: UploadFile = File(...)):
    """
    Endpoint pour scanner une CIN tunisienne.

    Args:
        file: Image de la CIN (JPEG, PNG, BMP, WebP)

    Returns:
        Données extraites et validées de la CIN
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier requis")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Extensions acceptées: {ALLOWED_EXTENSIONS}",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Fichier vide")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux (max: {MAX_FILE_SIZE // (1024*1024)}MB)",
        )

    temp_path = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        result = process_document(temp_path)
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Fichier image non valide")
    except Exception as e:
        logger.error(f"Erreur traitement OCR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=app_config.host,
        port=app_config.port,
        reload=app_config.debug,
    )
