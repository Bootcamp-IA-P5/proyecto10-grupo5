import streamlit as st
import requests
import pandas as pd
import os

# Get the backend URL from an environment variable or use a default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL = os.getenv("API_URL", BACKEND_URL)

# --- App Configuration ---
st.set_page_config(
    page_title="Toxicity Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- API Documentation Link ---
st.markdown(
    f"""
    <div style="text-align: right;">
        <a href="{API_URL}" target="_blank">API Documentation</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- App Title ---
st.title("Toxicity Classifier")

# --- Model Selection ---
# This dictionary makes it easy to add more models in the future.
# The key is the user-facing name, and the value is the endpoint path.
ENDPOINTS = {
    "Logistic Regression": "logistic_regression/predict",
    "Naive Bayes": "naive_bayes/predict",
}

selected_model = st.selectbox(
    "Choose a model for prediction:",
    options=list(ENDPOINTS.keys())
)

# --- Input Features ---
st.header("Enter Text for Analysis")
input_text = st.text_area(
    "Text to classify:", 
    "You should be banned from the internet for saying that.",
    height=150
)

# --- Prediction ---
if st.button("Get Prediction", type="primary"):
    endpoint_path = ENDPOINTS[selected_model]
    api_url = f"{BACKEND_URL}/{endpoint_path}"

    # The input data should match what your backend API expects: a JSON with a "text" key.
    input_data = {"text": input_text}

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(api_url, json=input_data)
            response.raise_for_status()  # Raise an exception for bad status codes

        prediction = response.json()

        st.success("Prediction successful!")
        
        # Create a DataFrame for better visualization
        df = pd.DataFrame(prediction.items(), columns=['Category', 'Probability'])
        
        # Display the results in a more readable format
        st.dataframe(df.set_index('Category'))

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")