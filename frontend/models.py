import requests
import pandas as pd
from config import BACKEND_URL

from utils.log_setup import log_setup
from utils.logger import Logger

log_setup("frontend")
logger = Logger()

# --- Model Definitions ---
# This dictionary makes it easy to add more models in the future.
# The key is the user-facing name, and the value is the endpoint path.
ENDPOINTS = {
    "Logistic Regression": "logistic_regression/predict",
    "Naive Bayes": "naive_bayes/predict",
    "SVM (Support Vector Machine)": "svm/predict",
    "XGBoost (TF-IDF)": "xgboost/predict",
}


def get_prediction(selected_model: str, input_text: str) -> pd.DataFrame | dict:
    """
    Sends a prediction request to the selected backend model endpoint.

    Args:
        selected_model: The user-facing name of the model (e.g., "Logistic Regression").
        input_text: The text string to be classified.

    Returns:
        A pandas DataFrame of categories and probabilities on success, or
        a dictionary with an 'error' message on failure.
    """
    if selected_model not in ENDPOINTS:
        message = f"Unknown model selected: {selected_model}"
        logger.log.error(message)
        return {"error": message}

    endpoint_path = ENDPOINTS[selected_model]
    api_url = f"{BACKEND_URL}/{endpoint_path}"
    input_data = {"text": input_text}

    try:
        logger.log.info(f"Sending prediction request to {api_url}")
        response = requests.post(api_url, json=input_data)
        response.raise_for_status()  # Raise an exception for bad status codes

        prediction = response.json()
        logger.log.debug(prediction)

        # Create a DataFrame for better visualization
        df = pd.DataFrame(prediction.items(), columns=["Category", "Probability"])
        return df

    except requests.exceptions.RequestException as e:
        message = f"Could not connect to the backend: {e}"
        logger.log.error(message)
        return {"error": message}
    except ValueError as e:
        message = f"Invalid JSON response from the backend: {e}"
        logger.log.error(message)
        return {"error": message}
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        logger.log.error(message)
        return {"error": message}
