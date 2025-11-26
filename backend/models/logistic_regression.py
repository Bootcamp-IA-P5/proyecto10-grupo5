from email import message
from venv import logger
from fastapi import APIRouter, HTTPException
from backend.utils.model_loader import load_model, load_vectorizer
from backend.schemas import TextIn, ToxicityOut, OUTPUT_CATEGORIES

from utils.log_setup import log_setup
from utils.logger import Logger

# Load the specific model for this router
LOGISTIC_REGRESSION_MODEL = load_model('logistic_regression.pkl')
LOGISTIC_REGRESSION_VECTORIZER = load_vectorizer('text_vectorizer.pkl')

log_setup()
logger = Logger()

# Create a FastAPI router for this model
router = APIRouter(
    prefix="/logistic_regression",
    tags=["Logistic Regression"],
    responses={404: {"description": "Not found"}},
)

@router.post("/predict", response_model=ToxicityOut)
def predict_lr(request: TextIn):
    if LOGISTIC_REGRESSION_MODEL is None:
        message = "Logistic Regression Model is unavailable."
        logger.log.error(message)
        raise HTTPException(
            status_code=503, 
            detail=message
        )
    
    # Check for vectorizer availability (should be loaded in model_loader.py)
    if LOGISTIC_REGRESSION_VECTORIZER is None:
        message = "Text Vectorizer is unavailable."
        logger.log.error(message)
        raise HTTPException(
            status_code=503, 
            detail=message
        )

    try:
        # Prepare input
        logger.log.debug("Predicting Logistic Regression...")
        data_to_predict = [request.text]
        vectorized_data = LOGISTIC_REGRESSION_VECTORIZER.transform(data_to_predict)
        
        # Run prediction
        # Use predict_proba for probabilities if available, or predict
        prediction_scores_2d = LOGISTIC_REGRESSION_MODEL.predict(vectorized_data) 
        
        # Process and Validate
        prediction_scores = prediction_scores_2d[0] 
        
        if len(prediction_scores) != len(OUTPUT_CATEGORIES):
            message = f"Model output dimension mismatch. Expected {len(OUTPUT_CATEGORIES)} scores, but got {len(prediction_scores)}."
            logger.log.error(message)
            raise ValueError(message)

        # Map scores and return
        response_data = dict(zip(OUTPUT_CATEGORIES, prediction_scores.astype(float)))
        logger.log.debug(f"Prediction result: {response_data}")
        return response_data

    except Exception as e:
        logger.log.error(f"LR Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal LR Model Error: {e}")