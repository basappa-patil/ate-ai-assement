"""ATE AI Engineering Assessment — core package."""

__version__ = "0.1.0"

from src.ate_ai.data import load_data
from src.ate_ai.cleaning import clean_data
from src.ate_ai.analytics import compute_yield_analytics

__all__ = ["load_data", "clean_data", "compute_yield_analytics", "__version__"]
