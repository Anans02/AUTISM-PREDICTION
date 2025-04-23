import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('autism_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page configuration
st.set_page_config(page_title="Autism Prediction", page_icon="🧠", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #F0F2F6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6em 1.2em;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
            color: white;
        }
        .stSlider {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 Autism Prediction App")
st.markdown("""
Welcome to the **Autism Spectrum Disorder (ASD)** prediction app.  
Answer the short questions below and click **Predict** to get the result.

*Note: This is for educational purposes only.*
""")

st.markdown("---")
st.subheader("Survey Responses (0 = No, 1 = Yes)")

# Split into two columns
col1, col2 = st.columns(2)

with col1:
    a1 = st.slider("A1: Makes eye contact", 0, 1)
    a2 = st.slider("A2: Enjoys social situations", 0, 1)
    a3 = st.slider("A3: Upset by small changes", 0, 1)
    a4 = st.slider("A4: Notices small sounds", 0, 1)
    a5 = st.slider("A5: Enjoys routines", 0, 1)
    a6 = st.slider("A6: Gets easily distracted", 0, 1)
    a7 = st.slider("A7: Prefers to be alone", 0, 1)
    a8 = st.slider("A8: Finds it hard to make friends", 0, 1)
    a9 = st.slider("A9: Responds to name", 0, 1)
    a10 = st.slider("A10: Shows empathy", 0, 1)

with col2:
    a11 = st.slider("A11: Sensitive to noise", 0, 1)
    a12 = st.slider("A12: Enjoys pretend play", 0, 1)
    a13 = st.slider("A13: Likes orderly routines", 0, 1)
    a14 = st.slider("A14: Makes unusual movements", 0, 1)
    a15 = st.slider("A15: Strong memory for facts", 0, 1)
    a16 = st.slider("A16: Avoids physical contact", 0, 1)
    a17 = st.slider("A17: Good attention span", 0, 1)
    a18 = st.slider("A18: Shows interest in others", 0, 1)
    a19 = st.slider("A19: Can follow instructions", 0, 1)
    a20 = st.slider("A20: Uses gestures", 0, 1)

# Collect input data
input_data = np.array([[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                        a11, a12, a13, a14, a15, a16, a17, a18, a19, a20]])

# Scale data
input_scaled = scaler.transform(input_data)

# Predict on button click
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ The person is **likely to have Autism Spectrum Disorder (ASD)**.")
    else:
        st.success("✅ The person is **unlikely to have Autism Spectrum Disorder (ASD)**.")
