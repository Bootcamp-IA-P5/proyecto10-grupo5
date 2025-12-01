from email import message
from fastapi import APIRouter, HTTPException
from backend.utils.model_loader import load_model, load_vectorizer
from backend.schemas import TextIn, ToxicityOut, OUTPUT_CATEGORIES

from utils.log_setup import log_setup
from utils.logger import Logger

# Load the specific model for this router
PREFIX = "test_"
NAIVE_BAYES_MODEL = load_model(PREFIX + 'naive_bayes.pkl') # Assumes this file exists
NAIVE_BAYES_VECTORIZER = load_vectorizer(PREFIX + 'text_vectorizer.pkl')

log_setup()
logger = Logger()

# Create a FastAPI router for this model
router = APIRouter(
    prefix="/naive_bayes",
    tags=["Naive Bayes"],
    responses={404: {"description": "Not found"}},
)

@router.post("/predict", response_model=ToxicityOut)
def predict_nb(request: TextIn):
    if NAIVE_BAYES_MODEL is None:
        message = "Naive Bayes Model is unavailable."
        logger.log.error(message)
        raise HTTPException(
            status_code=503, 
            detail=message
        )
    
    if NAIVE_BAYES_VECTORIZER is None:
        message = "Text Vectorizer is unavailable."
        logger.log.error("Text Vectorizer is unavailable.")
        raise HTTPException(
            status_code=503, 
            detail=message
        )

    try:
        # Prepare input
        logger.log.debug("Predicting Naïve Bayes...")
        data_to_predict = [request.text]
        vectorized_data = NAIVE_BAYES_VECTORIZER.transform(data_to_predict)
        
        # Run prediction
        # Use predict_proba for probabilities if available, or predict
        prediction_scores_2d = NAIVE_BAYES_MODEL.predict(vectorized_data) 
        
        # Process and Validate (similar to LR model)
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
        logger.log.error(f"NB Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal NB Model Error: {e}")