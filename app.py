import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AgriYield", page_icon="🌾")
st.title("🌾 AgriYield Viability Engine")
st.write("Determine crop survival probability using localized soil and weather telemetry.")

# --- ML Model Setup ---
@st.cache_resource
def build_agri_model():
    np.random.seed(123)
    n_samples = 400
    
    # generating soil and weather data
    ph_level = np.random.normal(6.5, 1.0, n_samples)
    rainfall_mm = np.random.normal(100, 40, n_samples)
    temp_c = np.random.normal(24, 6, n_samples)
    
    # Logic: Crop likes pH 6-7.5, rainfall 80-150mm, temp 20-30C
    # If conditions are good, survival = 1, else 0
    conditions_met = (
        (ph_level >= 5.5) & (ph_level <= 7.8) &
        (rainfall_mm >= 60) & (rainfall_mm <= 160) &
        (temp_c >= 18) & (temp_c <= 32)
    )
    
    # Adding a little randomness so the ML has to actually learn
    noise = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
    survival = np.where(conditions_met ^ noise, 1, 0) # XOR for noise
    
    agri_data = pd.DataFrame({'ph': ph_level, 'rain': rainfall_mm, 'temp': temp_c, 'survival': survival})
    
    # Logistic Regression for binary classification
    log_reg = LogisticRegression()
    log_reg.fit(agri_data[['ph', 'rain', 'temp']], agri_data['survival'])
    return log_reg

model = build_agri_model()

# --- UI Controls ---
st.sidebar.header("🌍 Environmental Sensors")
val_ph = st.sidebar.slider("Soil pH Level", 0.0, 14.0, 6.5, 0.1)
val_rain = st.sidebar.slider("Monthly Rainfall (mm)", 0, 300, 100)
val_temp = st.sidebar.slider("Average Temp (°C)", 0, 50, 24)

# --- Prediction Engine ---
test_input = pd.DataFrame({'ph': [val_ph], 'rain': [val_rain], 'temp': [val_temp]})
prediction = model.predict(test_input)[0]
probability = model.predict_proba(test_input)[0][1] * 100

st.divider()
st.subheader("Crop Viability Assessment")

if prediction == 1:
    st.success(f"🌱 **Viable:** {probability:.1f}% probability of a successful harvest.")
else:
    st.error(f"⚠️ **High Risk of Failure:** Only {probability:.1f}% probability of survival.")
    
    # Basic logic checks to tell the user WHY it failed
    if val_ph < 5.5 or val_ph > 7.8:
        st.caption("- Issue detected: Soil pH is outside optimal range (5.5 - 7.8).")
    if val_rain < 60:
        st.caption("- Issue detected: Insufficient rainfall for this crop variant.")
