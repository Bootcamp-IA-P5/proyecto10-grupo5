import joblib
import os
from typing import Any

from utils.logger import Logger
from utils.log_setup import log_setup

# --- Model Artifact Paths (Define them once) ---
BASE_MODEL_DIR = 'resources/models'

log_setup()
logger = Logger()

def load_vectorizer(vectorizer_filename: str) -> Any:
    """Loads the shared text vectorizer."""
    try:
        vectorizer_path = os.path.join(BASE_MODEL_DIR, vectorizer_filename)
        vectorizer = joblib.load(vectorizer_path)
        logger.log.info(f"Vectorizer loaded successfully from {vectorizer_path}.")
        return vectorizer
    except FileNotFoundError:
        logger.log.error(f"Vectorizer file not found at {vectorizer_path}!")
        raise RuntimeError("Vectorizer file not found.")
    except Exception as e:
        logger.log.error(f"Failed to load vectorizer: {e}")
        raise

def load_model(model_filename: str) -> Any:
    """Loads a specific model file."""
    model_path = os.path.join(BASE_MODEL_DIR, model_filename)
    try:
        model = joblib.load(model_path)
        logger.log.info(f"Model '{model_filename}' loaded successfully.")
        return model
    except FileNotFoundError:
        logger.log.error(f"Model file not found at {model_path}!")
        # Allows the server to start, but the endpoint using this model will fail.
        # For production, you might raise a hard exception here (like in your original code).
        return None
    except Exception as e:
        logger.log.error(f"Failed to load model '{model_filename}': {e}")
        return None
