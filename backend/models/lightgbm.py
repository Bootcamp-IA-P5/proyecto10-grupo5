from fastapi import APIRouter, HTTPException
from backend.utils.model_loader import load_model, load_vectorizer
from backend.schemas import TextIn, ToxicityOut, OUTPUT_CATEGORIES

from utils.log_setup import log_setup
from utils.logger import Logger

from backend.config import PREFIX

# Load the specific model for this router
LIGHTGBM_MODEL = load_model(PREFIX + "model-LightGBM-original.pkl")
LIGHTGBM_VECTORIZER = load_vectorizer(PREFIX + "lightgbm_tfidf_vectorizer.pkl")

log_setup()
logger = Logger()

# Create a FastAPI router for this model
router = APIRouter(
    prefix="/lightgbm",
    tags=["LightGBM"],
    responses={404: {"description": "Not found"}},
)


@router.post("/predict", response_model=ToxicityOut)
def predict_lightgbm(request: TextIn):
    if LIGHTGBM_MODEL is None:
        message = "LightGBM Model is unavailable."
        logger.log.error(message)
        raise HTTPException(status_code=503, detail=message)

    if LIGHTGBM_VECTORIZER is None:
        message = "TF-IDF Vectorizer is unavailable."
        logger.log.error(message)
        raise HTTPException(status_code=503, detail=message)

    try:
        # Prepare input
        logger.log.debug("Predicting LightGBM...")
        data_to_predict = [request.text]
        vectorized_data = LIGHTGBM_VECTORIZER.transform(data_to_predict)

        # Run prediction
        # LightGBM returns binary predictions (0 or 1) for multi-label classification
        prediction_scores_2d = LIGHTGBM_MODEL.predict(vectorized_data)

        # Process and Validate
        prediction_scores = prediction_scores_2d[0]

        if len(prediction_scores) != len(OUTPUT_CATEGORIES):
            message = f"Model output dimension mismatch. Expected {len(OUTPUT_CATEGORIES)} scores, but got {len(prediction_scores)}."
            logger.log.error(message)
            raise ValueError(message)

        # Map scores and return
        # Convert binary predictions to float for consistency with other models
        response_data = dict(zip(OUTPUT_CATEGORIES, prediction_scores.astype(float)))

        # Ensure IsToxic is 1 if ANY other toxicity category is detected
        # This handles cases where the model detects sub-categories but not IsToxic
        sub_categories = [
            "IsAbusive",
            "IsProvocative",
            "IsObscene",
            "IsHatespeech",
            "IsRacist",
        ]
        if any(response_data.get(cat, 0) == 1.0 for cat in sub_categories):
            response_data["IsToxic"] = 1.0

        logger.log.debug(f"Prediction result: {response_data}")
        return response_data

    except Exception as e:
        logger.log.error(f"LightGBM Prediction Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal LightGBM Model Error: {e}"
        )
