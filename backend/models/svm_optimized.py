from fastapi import APIRouter, HTTPException
from backend.utils.model_loader import load_model, load_vectorizer
from backend.schemas import TextIn, ToxicityOut, OUTPUT_CATEGORIES
import joblib
import os

from utils.log_setup import log_setup
from utils.logger import Logger

# Load the SVM optimized model
SVM_MODEL = load_model("svm_toxicity_optimized_20251128_145713.pkl")
SVM_VECTORIZER = load_vectorizer("tfidf_vectorizer_20251128_145713.pkl")

# Load label encoder
try:
    label_encoder_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "resources",
        "models",
        "label_encoder_20251128_145713.pkl",
    )
    LABEL_ENCODER = joblib.load(label_encoder_path)
except Exception as e:
    print(f"Warning: Could not load label encoder: {e}")
    LABEL_ENCODER = None

log_setup()
logger = Logger()

# Create a FastAPI router for this model
router = APIRouter(
    prefix="/svm_optimized",
    tags=["SVM Optimized"],
    responses={404: {"description": "Not found"}},
)


@router.post("/predict", response_model=ToxicityOut)
def predict_svm_optimized(request: TextIn):
    if SVM_MODEL is None:
        message = "SVM Optimized Model is unavailable."
        logger.log.error(message)
        raise HTTPException(status_code=503, detail=message)

    if SVM_VECTORIZER is None:
        message = "TF-IDF Vectorizer is unavailable."
        logger.log.error(message)
        raise HTTPException(status_code=503, detail=message)

    try:
        # Prepare input
        logger.log.debug("Predicting with SVM Optimized...")
        data_to_predict = [request.text]
        vectorized_data = SVM_VECTORIZER.transform(data_to_predict)

        # Run prediction
        # SVM model doesn't have probability=True, so we use predict directly
        prediction = SVM_MODEL.predict(vectorized_data)[0]

        # Get predicted class
        predicted_class = (
            LABEL_ENCODER.classes_[prediction]
            if LABEL_ENCODER
            else f"Class_{prediction}"
        )

        logger.log.info(f"Predicted class: {predicted_class}")

        # Map to OUTPUT_CATEGORIES format
        # Initialize all categories to 0.0
        response_data = {cat: 0.0 for cat in OUTPUT_CATEGORIES}

        # Map the predicted category to the appropriate output format
        category_mapping = {
            "Abusive": "IsAbusive",
            "Hatespeech": "IsHatespeech",
            "NonToxic": "IsToxic",  # NonToxic means IsToxic=0
            "Racist": "IsRacist",
            "Obscene": "IsObscene",
            "Provocative": "IsProvocative",
        }

        # Set IsToxic based on prediction
        if predicted_class == "NonToxic":
            response_data["IsToxic"] = 0.0
        else:
            response_data["IsToxic"] = 1.0
            # Set the specific category
            if predicted_class in category_mapping:
                mapped_category = category_mapping[predicted_class]
                if mapped_category in response_data:
                    response_data[mapped_category] = 1.0

        logger.log.debug(f"Prediction result: {response_data}")
        return response_data

    except Exception as e:
        logger.log.error(f"SVM Optimized Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal SVM Model Error: {e}")
