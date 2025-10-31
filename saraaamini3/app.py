"""
Streamlit Frontend for Phishing URL Detection
"""

import streamlit as st
import os
import pandas as pd
from random_forest_model import RandomForestPhishingDetector
from xgboost_model import XGBoostPhishingDetector


st.set_page_config(
    page_title="URL Risk Analyzer",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ URL Risk Analyzer")
st.caption("Analyze a URL using Random Forest or XGBoost models")


# Load models
@st.cache_resource
def load_model(model_type):
    """Load the selected model"""
    models_dir = 'models'
    
    if model_type == "Random Forest":
        model_path = os.path.join(models_dir, 'random_forest_model.pkl')
        if os.path.exists(model_path):
            return RandomForestPhishingDetector.load(model_path), "Random Forest"
        else:
            return None, "Random Forest"
    else:  # XGBoost
        model_path = os.path.join(models_dir, 'xgboost_model.pkl')
        if os.path.exists(model_path):
            return XGBoostPhishingDetector.load(model_path), "XGBoost"
        else:
            return None, "XGBoost"


def render_result_card(title: str, result: dict):
    """Render result card similar to the provided style"""
    with st.container():
        st.subheader(title)
        if not result:
            st.info("No result returned.")
            return
        if result.get("error"):
            st.error(result["error"])
            return

        pred = result.get("prediction") or ("LEGITIMATE" if result.get("is_safe") else "PHISHING")
        risk = result.get("risk_score", result.get("final_confidence", 0.0))
        flags = result.get("red_flag_count", len(result.get("red_flags", [])))
        method = result.get("decision_method", "")

        cols = st.columns(3)
        cols[0].metric("Prediction", "SAFE" if pred == "LEGITIMATE" else "UNSAFE")
        cols[1].metric("Risk %", f"{float(risk):.1f}%")
        cols[2].metric("Red Flags", f"{int(flags)}")
        if method:
            st.caption(f"Method: {method}")

        # Show confidence probabilities
        if result.get("confidence"):
            conf_col1, conf_col2 = st.columns(2)
            with conf_col1:
                st.caption(f"Legitimate: {result['confidence'].get('legitimate', 0)*100:.2f}%")
            with conf_col2:
                st.caption(f"Phishing: {result['confidence'].get('phishing', 0)*100:.2f}%")

        # Show red flags
        if result.get("red_flags"):
            st.markdown("**Triggered Red Flags**")
            st.write("\n".join([f"- {flag}" for flag in result["red_flags"]]))
        else:
            st.caption("No red flags detected.")


# Form for input
with st.form("analyzer_form"):
    url = st.text_input("Enter URL", placeholder="https://example.com/login")
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Both"], index=0)
    submitted = st.form_submit_button("Analyze")


if submitted:
    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        # Normalize URL
        url_input = url.strip()
        if not url_input.startswith(('http://', 'https://')):
            url_input = f"https://{url_input}"

        with st.spinner("Analyzing URL..."):
            try:
                results = {}
                
                if model_choice == "Both":
                    # Load both models
                    rf_model, _ = load_model("Random Forest")
                    xgb_model, _ = load_model("XGBoost")
                    
                    if rf_model is None or xgb_model is None:
                        st.error("One or more models not found. Please run `python train_models.py` first.")
                        st.stop()
                    
                    # Get predictions from both
                    results['rf'] = rf_model.predict_url(url_input, verbose=False)
                    results['xgb'] = xgb_model.predict_url(url_input, verbose=False)
                    
                    # Display results in tabs
                    tab1, tab2 = st.tabs(["Random Forest", "XGBoost"])
                    with tab1:
                        render_result_card("Random Forest Result", results.get("rf"))
                    with tab2:
                        render_result_card("XGBoost Result", results.get("xgb"))
                    
                elif model_choice == "Random Forest":
                    model, model_name = load_model("Random Forest")
                    if model is None:
                        st.error("Random Forest model not found. Please run `python train_models.py` first.")
                        st.stop()
                    result = model.predict_url(url_input, verbose=False)
                    render_result_card(f"{model_name} Result", result)
                    
                else:  # XGBoost
                    model, model_name = load_model("XGBoost")
                    if model is None:
                        st.error("XGBoost model not found. Please run `python train_models.py` first.")
                        st.stop()
                    result = model.predict_url(url_input, verbose=False)
                    render_result_card(f"{model_name} Result", result)
                    
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.exception(e)

st.divider()
st.caption("Models expected in project-root 'models/' directory. Run `python train_models.py` to train and save models.")

