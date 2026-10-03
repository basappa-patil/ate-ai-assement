"""Project-wide path constants and configuration."""

from pathlib import Path

# src/ate_ai/config.py → parent.parent.parent = project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

DEFAULT_CSV: Path = DATA_RAW / "ATE_AI_Engineering_Assessment.csv"

RANDOM_SEED: int = 42
