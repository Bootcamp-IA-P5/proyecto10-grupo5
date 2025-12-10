"""
XGBoost Multi-Label Toxicity Classification Model API
Provides predictions across 6 toxicity categories using TF-IDF features.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
from xgboost import XGBClassifier
import os
from pathlib import Path

from utils.log_setup import log_setup
from utils.logger import Logger

# Initialize logging
log_setup("backend")
logger = Logger()

# --- Request/Response Models ---
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    IsToxic: float
    IsAbusive: float
    IsProvocative: float
    IsObscene: float
    IsHatespeech: float
    IsRacist: float

# --- Router Setup ---
router = APIRouter(prefix="/xgboost", tags=["XGBoost Model"])

# --- Model Loading ---
# Define paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "resources" / "models"
JSON_DIR = BASE_DIR / "eda" / "json"

# Target categories in the order they should be processed
TARGET_CATEGORIES = [
    'IsToxic',
    'IsAbusive', 
    'IsProvocative',
    'IsObscene',
    'IsHatespeech',
    'IsRacist'
]

# Global variables for models and vectorizer
tfidf_vectorizer = None
xgboost_models = {}

def load_xgboost_resources():
    """
    Load the TF-IDF vectorizer and all 6 XGBoost models on startup.
    """
    global tfidf_vectorizer, xgboost_models
    
    try:
        # Load TF-IDF Vectorizer
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        logger.log.info(f"Loading TF-IDF vectorizer from {vectorizer_path}")
        
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        tfidf_vectorizer = joblib.load(vectorizer_path)
        logger.log.info("TF-IDF vectorizer loaded successfully.")
        
        # Load all XGBoost models
        for target in TARGET_CATEGORIES:
            model_path = JSON_DIR / f"xgb_tfidf_model_{target}.json"
            logger.log.info(f"Loading XGBoost model for {target} from {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = XGBClassifier()
            model.load_model(str(model_path))
            xgboost_models[target] = model
            logger.log.info(f"Model '{target}' loaded successfully.")
        
        logger.log.info("All XGBoost models loaded successfully.")
        
    except Exception as e:
        logger.log.error(f"Error loading XGBoost resources: {e}")
        raise RuntimeError(f"Failed to load XGBoost models: {e}")

# Load models when module is imported
load_xgboost_resources()

# --- Prediction Endpoint ---
@router.post("/predict", response_model=PredictionResponse)
def predict_toxicity(request: PredictionRequest):
    """
    Predict toxicity across 6 categories for the given text.
    
    Args:
        request: PredictionRequest containing the text to classify
        
    Returns:
        PredictionResponse with probabilities for each toxicity category
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        input_text = request.text
        logger.log.info(f"Received prediction request for text: '{input_text[:50]}...'")
        
        # Validate input
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        # Transform text using TF-IDF vectorizer
        logger.log.debug("Transforming text with TF-IDF vectorizer...")
        text_features = tfidf_vectorizer.transform([input_text])
        
        # Get predictions from all models
        predictions = {}
        for target in TARGET_CATEGORIES:
            model = xgboost_models[target]
            
            # Get probability prediction (probability of positive class)
            proba = model.predict_proba(text_features)[0][1]
            predictions[target] = round(float(proba), 4)
            
            logger.log.debug(f"{target}: {predictions[target]}")
        
        logger.log.info("Prediction completed successfully.")
        return PredictionResponse(**predictions)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- Health Check Endpoint ---
@router.get("/health")
def health_check():
    """
    Check if all XGBoost models are loaded and ready.
    """
    models_loaded = len(xgboost_models) == len(TARGET_CATEGORIES)
    vectorizer_loaded = tfidf_vectorizer is not None
    
    status = {
        "status": "healthy" if (models_loaded and vectorizer_loaded) else "unhealthy",
        "vectorizer_loaded": vectorizer_loaded,
        "models_loaded": list(xgboost_models.keys()),
        "models_count": len(xgboost_models),
        "expected_models": len(TARGET_CATEGORIES)
    }
    
    if not models_loaded or not vectorizer_loaded:
        raise HTTPException(status_code=503, detail=status)
    
    return status