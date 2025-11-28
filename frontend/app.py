# Get the path to the current script's directory (frontend/)
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory (the root folder)
root_dir = os.path.join(current_dir, os.pardir)

# Insert the root directory into the Python path
sys.path.insert(0, root_dir)

import streamlit as st
import pandas as pd
from frontend.config import API_URL, APP_TITLE, PAGE_ICON
from frontend.models import ENDPOINTS, get_prediction 

# --- App Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded",
)

def render_sidebar():
    """Renders all elements within the sidebar."""

    # --- Top Section ---
    st.sidebar.title(f"{PAGE_ICON} {APP_TITLE}")

    # --- Model Selection ---
    st.sidebar.subheader("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose an AI model for classification:",
        options=list(ENDPOINTS.keys()),
        key="model_selector"
    )

    # --- API Documentation section ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div>
            <p style='font-size: small; color: #555; margin-bottom: 8px;'>API Integration Details</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # --- API Documentation Button  ---
    st.sidebar.link_button(
        label="🔗 View API Docs", 
        url=API_URL, 
        type="secondary",
        help="Opens the backend API documentation in a new tab."
    )
    
    
    return selected_model

def generate_summary(df: pd.DataFrame) -> str:
    """
    Analyzes the prediction DataFrame and generates a concise summary sentence.
    """
    # Using a threshold (e.g., > 0.9999) is safer than exactly 1.0 for float comparisons
    threshold = 0.9999
    
    # Filter the categories that meet the threshold
    toxic_categories = df[df['Probability'] > threshold]['Category'].tolist()
    
    # Check if 'IsToxic' is one of the detected categories
    is_toxic_detected = 'IsToxic' in toxic_categories
    
    # The categories we want to list (excluding 'IsToxic' for cleaner phrasing)
    sub_categories = [cat for cat in toxic_categories if cat != 'IsToxic']

    # 1. Handle the non-toxic case
    if not is_toxic_detected:
        return "The text is classified as **non-toxic**."
    
    # 2. Prepare sub-category names for clean display (e.g., 'IsHateSpeech' -> 'hate speech')
    categories_clean = [cat.replace('Is', '').replace('Hate', ' hate').lower() for cat in sub_categories]
    
    # 3. Handle the case where ONLY 'IsToxic' is true
    if not categories_clean:
        return "The text is classified as **toxic**, but does not clearly fall into a specific sub-category."
    
    # 4. Handle toxic + one or more sub-categories
    
    # Start the sentence
    summary = "The text is **toxic**"
    
    # Join the rest of the categories
    if len(categories_clean) == 1:
        # e.g., "...and classified as provocative."
        summary += f", and is specifically characterized as **{categories_clean[0]}**."
    else:
        # e.g., "...and contains characteristics of abusive, racist, and religious hate."
        last_category = categories_clean[-1]
        other_categories = categories_clean[:-1]
        
        # Build the descriptive list
        category_list = ", ".join([f"**{c}**" for c in other_categories])
        summary += f", containing characteristics of {category_list}, and **{last_category}**."
        
    return summary

def render_main_content(selected_model):
    """Renders the input and results area in the main panel."""
    
    st.title("Toxicity Analysis Tool")
    st.markdown(
        f"Use the **{selected_model}** model to analyze the probability of different toxicity categories."
    )
    st.markdown("---")

    st.header("1. Enter Text for Analysis")
    input_text = st.text_area(
        "Text to classify:", 
        "You should be banned from the internet for saying that.",
        height=75,
        key="input_text_area"
    )

    st.header("2. Get Prediction")
    if st.button("Analyze Text", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
            return

        with st.spinner(f"Classifying text using **{selected_model}**..."):
            result = get_prediction(selected_model, input_text)

        st.subheader("3. Prediction Results")

        if isinstance(result, pd.DataFrame):
            st.success("Analysis successful!")
            
            # --- NEW SUMMARY DISPLAY ---
            summary_text = generate_summary(result)
            
            # Display the human-readable summary
            st.markdown(f"""
            <div style="
                border: 2px solid #e0e0e0; 
                padding: 5px; 
                border-radius: 8px; 
                background-color: #f7f7f7;
                font-size: 1.1em;
            ">
            {summary_text}
            </div>
            """, unsafe_allow_html=True)
            # --- END NEW SUMMARY DISPLAY ---
            
            # Optional: Keep a small table for full transparency
            with st.expander("Show detailed probabilities"):
                st.dataframe(
                    result.set_index('Category').style.format({"Probability": "{:.4f}"}), 
                    use_container_width=True
                )
            
        elif isinstance(result, dict) and 'error' in result:
            st.error(result['error'])
            st.exception(result['error'])

def run_app():
    """Main function to run the Streamlit application."""
    selected_model = render_sidebar()
    render_main_content(selected_model)

if __name__ == "__main__":
    run_app()