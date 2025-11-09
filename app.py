import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing components
model = joblib.load("Price_predictor_model")
encoding_maps = joblib.load("encoding_maps")
scaler = joblib.load("feature_scaler")

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Prediction App")
st.write("Enter laptop details below to estimate its price (in Euros).")

# Input fields
product = st.text_input("Product Name", "MacBook Pro")
cpu_model = st.text_input("CPU Model", "Intel Core i5")
gpu_model = st.text_input("GPU Model", "NVIDIA GeForce")
ram = st.number_input("RAM (GB)", min_value=2, max_value=64, step=2, value=8)
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=5.0, step=0.1, value=2.3)
primary_storage = st.number_input("Primary Storage (GB)", min_value=64, max_value=2048, step=64, value=512)
typename = st.selectbox("Laptop Type", ['Ultrabook', 'Gaming', 'Notebook', 'Workstation', '2 in 1 Convertible', 'Netbook'])

# Prediction logic
if st.button("Predict Price"):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame({
            'Product': [product],
            'CPU_model': [cpu_model],
            'GPU_model': [gpu_model],
            'Ram': [ram],
            'CPU_freq': [cpu_freq],
            'PrimaryStorage': [primary_storage],
            'TypeName': [typename]
        })

        # Apply target encoding using stored dictionaries
        for col in ['Product', 'CPU_model', 'GPU_model', 'TypeName']:
            mapping = encoding_maps.get(col, {})
            input_df[col] = input_df[col].map(mapping)
            # Handle unseen categories
            if pd.isnull(input_df[col][0]):
                input_df[col] = np.mean(list(mapping.values())) if mapping else 0

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Predict
        predicted_price = model.predict(scaled_input)[0]
        st.success(f"💰 Predicted Laptop Price: €{predicted_price:,.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
