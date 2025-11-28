import requests
import pandas as pd
from config import BACKEND_URL

# --- Model Definitions ---
# This dictionary makes it easy to add more models in the future.
# The key is the user-facing name, and the value is the endpoint path.
ENDPOINTS = {
    "Logistic Regression": "logistic_regression/predict",
    "Naive Bayes": "naive_bayes/predict",
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
        return {"error": f"Unknown model selected: {selected_model}"}

    endpoint_path = ENDPOINTS[selected_model]
    api_url = f"{BACKEND_URL}/{endpoint_path}"
    input_data = {"text": input_text}

    try:
        response = requests.post(api_url, json=input_data)
        response.raise_for_status()  # Raise an exception for bad status codes

        prediction = response.json()
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame(prediction.items(), columns=['Category', 'Probability'])
        return df

    except requests.exceptions.RequestException as e:
        return {"error": f"Could not connect to the backend: {e}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}