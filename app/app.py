import streamlit as st
from src.prediction_pipeline import FraudDetectionPipeline
import pandas as pd
import yaml
import os
from PIL import Image

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💰",
    layout="wide"
)

# Load config
@st.cache_resource
def load_config():
    with open('config/params.yaml') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_pipeline():
    try:
        return FraudDetectionPipeline()
    except FileNotFoundError as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Initialize
config = load_config()
pipeline = load_pipeline()
default_threshold = config.get('prediction_threshold', 0.5)

# Sidebar
st.sidebar.header("System Configuration")
threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(default_threshold),
    step=0.05,
    help="Higher values reduce false positives but may miss some fraud"
)

st.sidebar.markdown("---")
st.sidebar.header("Model Performance")
if 'metrics' in config:
    st.sidebar.metric("F1 Score", f"{config['metrics']['f1_score']:.3f}")
    st.sidebar.metric("ROC-AUC", f"{config['metrics']['roc_auc']:.3f}")

# Main interface
st.title("💰 Real-time Fraud Detection System")
st.markdown("""
Detect potentially fraudulent transactions using machine learning.
""")

tab1, tab2 = st.tabs(["Single Transaction", "Batch Processing"])

with tab1:
    with st.form("single_transaction_form"):
        st.header("Transaction Details")
        
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (USD)", min_value=0.0, value=100.0, step=0.01)
            time = st.number_input("Time (seconds)", min_value=0, value=0)
        with col2:
            v1 = st.number_input("V1 (PCA Component)", value=0.0)
            v2 = st.number_input("V2 (PCA Component)", value=0.0)
        
        v3 = st.number_input("V3 (PCA Component)", value=0.0)
        v4 = st.number_input("V4 (PCA Component)", value=0.0)
        
        submitted = st.form_submit_button("🔍 Check for Fraud")
        
        if submitted:
            input_data = {
                'Time': time,
                'V1': v1,
                'V2': v2,
                'V3': v3,
                'V4': v4,
                'Amount': amount
            }
            
            with st.spinner("Analyzing transaction..."):
                try:
                    result = pipeline.predict(input_data, threshold)
                    
                    st.subheader("Prediction Results")
                    if result['is_fraud']:
                        st.error(f"🚨 Fraud Detected! (Probability: {result['probability']:.2%})")
                        st.warning("This transaction has been flagged as potentially fraudulent.")
                    else:
                        st.success(f"✅ Legitimate Transaction (Probability: {result['probability']:.2%})")
                    
                    # Show probability gauge
                    st.progress(result['probability'])
                    st.caption(f"Fraud Probability: {result['probability']:.2%}")
                    
                    # Show feature values
                    with st.expander("View feature details"):
                        st.json(result['features'])
                        
                except ValueError as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.header("Batch Processing")
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Process Batch"):
                with st.spinner(f"Processing {len(df)} transactions..."):
                    results = pipeline.batch_predict(df, threshold)
                    
                    st.success("Batch processing complete!")
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name='fraud_predictions.csv',
                        mime='text/csv'
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.caption("Fraud Detection System v1.0 | For demonstration purposes only")