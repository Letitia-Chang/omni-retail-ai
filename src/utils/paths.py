from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "hm"

MODEL_DIR = ROOT_DIR / "saved_models"

SEGMENTATION_MODEL_DIR = MODEL_DIR / "segmentation"
PURCHASE_MODEL_DIR = MODEL_DIR / "purchase_model"

REPORTS_DIR = ROOT_DIR / "reports"