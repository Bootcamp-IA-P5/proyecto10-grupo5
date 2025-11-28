import streamlit as st
from config import API_URL, APP_TITLE, PAGE_ICON
from models import ENDPOINTS, get_prediction

# --- App Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded",
)

def render_api_doc_link():
    """Renders the API documentation link in the top right."""
    st.markdown(
        f"""
        <div style="text-align: right;">
            <a href="{API_URL}" target="_blank">API Documentation</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_model_selection():
    """Renders the model selection dropdown."""
    st.title(APP_TITLE)
    selected_model = st.selectbox(
        "Choose a model for prediction:",
        options=list(ENDPOINTS.keys())
    )
    return selected_model

def render_input_area():
    """Renders the text input area."""
    st.header("Enter Text for Analysis")
    input_text = st.text_area(
        "Text to classify:", 
        "You should be banned from the internet for saying that.",
        height=150
    )
    return input_text

def run_app():
    """Main function to run the Streamlit application."""
    render_api_doc_link()
    
    selected_model = render_model_selection()
    input_text = render_input_area()

    # --- Prediction Button ---
    if st.button("Get Prediction", type="primary"):
        with st.spinner(f"Getting prediction from **{selected_model}**..."):
            result = get_prediction(selected_model, input_text)

        if isinstance(result, pd.DataFrame):
            st.success("Prediction successful!")
            # Display the results in a readable format
            st.dataframe(result.set_index('Category'))
        elif isinstance(result, dict) and 'error' in result:
            st.error(result['error'])
        else:
            st.error("An unexpected error occurred during prediction.")

if __name__ == "__main__":
    import pandas as pd # Import here to ensure get_prediction can be tested standalone
    run_app()