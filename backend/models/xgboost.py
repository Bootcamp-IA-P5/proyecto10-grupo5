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
import re
from typing import List
import spacy
from collections import Counter
from itertools import chain

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
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "resources" / "models"
JSON_DIR = BASE_DIR / "eda" / "json"

TARGET_CATEGORIES = [
    'IsToxic',
    'IsAbusive', 
    'IsProvocative',
    'IsObscene',
    'IsHatespeech',
    'IsRacist'
]

# Global variables
tfidf_vectorizer = None
xgboost_models = {}
nlp = None
high_frequency_stopwords = set()

# Top 100 high-frequency words that were removed during training
# CRITICAL: These must match exactly what you used in training
HIGH_FREQ_WORDS = {
    'black', 'people', 'white', 't', 's', 'police', 'shoot', 'like', 'cop', 'brown',
    'man', 'video', 'know', 'gun', 'say', 'really', 'time', 'kill', 'want', 'year',
    'think', 'look', 'get', 'make', 'tell', 'take', 'use', 'find', 'work', 'give',
    'come', 'thing', 'need', 'way', 'right', 'problem', 'officer', 'actually', 'believe',
    'country', 'fact', 'case', 'call', 'day', 'feel', 'woman', 'u', 'leave', 'question',
    'matter', 'point', 'law', 'try', 'happen', 'fuck', 'guy', 'live', 'watch', 'mean',
    'kid', 'world', 'lot', 'person', 'start', 'good', 'bad', 'shit', 'stop', 'life',
    'talk', 'understand', 'run', 'cause', 'racist', 'pretty', 'yes', 'real', 'place',
    'hand', 'reason', 'mind', 'read', 'turn', 'child', 'allow', 'force', 'sense', 'guy',
    'shoot', 'violence', 'care', 'number', 'situation', 'fucking', 'idiot', 'fuck', 'shit'
}

def load_xgboost_resources():
    """Load TF-IDF vectorizer, spaCy model, and all XGBoost models."""
    global tfidf_vectorizer, xgboost_models, nlp, high_frequency_stopwords
    
    try:
        # Load spaCy model
        logger.log.info("Loading spaCy model (en_core_web_sm)...")
        nlp = spacy.load('en_core_web_sm', disable=['parser'])
        logger.log.info("spaCy model loaded successfully.")
        
        # Set high-frequency stopwords
        high_frequency_stopwords = HIGH_FREQ_WORDS
        
        # Load TF-IDF Vectorizer (This path is already correct from previous fix)
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer-fixed.pkl"  
        logger.log.info(f"Loading TF-IDF vectorizer from {vectorizer_path}")
        
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        tfidf_vectorizer = joblib.load(vectorizer_path)
        logger.log.info("TF-IDF vectorizer loaded successfully.")
        
        # Load all XGBoost models
        for target in TARGET_CATEGORIES:
            # FIX IS HERE: Added the 'fixed-' prefix
            model_path = JSON_DIR / f"fixed-xgb_tfidf_model_{target}.json" 
            logger.log.info(f"Loading XGBoost model for {target} from {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = XGBClassifier()
            model.load_model(str(model_path))
            xgboost_models[target] = model
            logger.log.info(f"Model '{target}' loaded successfully.")
        
        logger.log.info("All XGBoost resources loaded successfully.")
        
    except Exception as e:
        logger.log.error(f"Error loading XGBoost resources: {e}")
        raise RuntimeError(f"Failed to load XGBoost models: {e}")

# Load models when module is imported
load_xgboost_resources()

# --- Text Preprocessing Pipeline ---
# This replicates EXACTLY what was done during training

def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = r'https?://\S+|www\.\S+|\S+\.(?:com|org|net|edu|gov|io|co|ai|ly)\S*'
    return re.sub(url_pattern, '', text)

def remove_special_chars(text: str) -> str:
    """Remove special characters, keep basic punctuation."""
    special_char_pattern = re.compile(r'[^\w\s\'"!?.,-]')
    cleaned = special_char_pattern.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    Process text: tokenize, lemmatize, remove stop words.
    This MUST match your training preprocessing exactly!
    """
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_.lower())
    
    return tokens

def remove_high_frequency_words(tokens: List[str]) -> List[str]:
    """Filter out the top 100 high-frequency words."""
    return [token for token in tokens if token not in high_frequency_stopwords]

def preprocess_text(text: str) -> str:
    """
    Apply the complete preprocessing pipeline used during training.
    
    Pipeline:
    1. Remove URLs
    2. Remove special characters
    3. Lowercase
    4. Tokenize and lemmatize with spaCy
    5. Remove stop words
    6. Join tokens back into string
    """
    # Step 1-3: Basic cleaning
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = text.lower()
    
    # Step 4-5: Tokenize and lemmatize
    tokens = tokenize_and_lemmatize(text)
    
    # Step 6: Remove high-frequency words
    tokens = remove_high_frequency_words(tokens)
    
    # Step 6: Join back into string for TF-IDF
    cleaned_text = ' '.join(tokens)
    
    logger.log.debug(f"Original text: '{text[:50]}...'")
    logger.log.debug(f"Preprocessed text: '{cleaned_text[:50]}...'")
    logger.log.debug(f"Token count: {len(tokens)}")
    
    return cleaned_text

# --- Prediction Endpoint ---
@router.post("/predict", response_model=PredictionResponse)
def predict_toxicity(request: PredictionRequest):
    """
    Predict toxicity across 6 categories for the given text.
    """
    try:
        input_text = request.text
        logger.log.info(f"Received prediction request for text: '{input_text[:50]}...'")
        
        # Validate input
        if not input_text or not input_text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        # CRITICAL: Apply the same preprocessing as training
        logger.log.debug("Preprocessing text with full pipeline...")
        preprocessed_text = preprocess_text(input_text)
        
        # Check if preprocessing resulted in empty text
        if not preprocessed_text or not preprocessed_text.strip():
            logger.log.warning("Preprocessing resulted in empty text - returning low probabilities")
            # Return low probabilities for all categories
            predictions = {target: 0.05 for target in TARGET_CATEGORIES}
            return PredictionResponse(**predictions)
        
        # Transform text using TF-IDF vectorizer
        logger.log.debug("Transforming text with TF-IDF vectorizer...")
        text_features = tfidf_vectorizer.transform([preprocessed_text])
        
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
    """Check if all XGBoost models are loaded and ready."""
    models_loaded = len(xgboost_models) == len(TARGET_CATEGORIES)
    vectorizer_loaded = tfidf_vectorizer is not None
    spacy_loaded = nlp is not None
    
    status = {
        "status": "healthy" if (models_loaded and vectorizer_loaded and spacy_loaded) else "unhealthy",
        "vectorizer_loaded": vectorizer_loaded,
        "spacy_loaded": spacy_loaded,
        "models_loaded": list(xgboost_models.keys()),
        "models_count": len(xgboost_models),
        "expected_models": len(TARGET_CATEGORIES),
        "high_freq_words_count": len(high_frequency_stopwords)
    }
    
    if not models_loaded or not vectorizer_loaded or not spacy_loaded:
        raise HTTPException(status_code=503, detail=status)
    
    return status