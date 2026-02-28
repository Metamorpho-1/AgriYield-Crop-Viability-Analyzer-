import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AgriYield", page_icon="🌾")
st.title("🌾 AgriYield Viability Engine")
st.write("Determine crop survival probability using localized soil and weather telemetry.")

# --- ML Model Setup ---
@st.cache_resource
def build_agri_model():
    np.random.seed(123)
    n_samples = 2000 # Increased dataset size
    
    # Generating WIDE data including extreme out-of-bounds conditions
    ph_level = np.random.uniform(0.0, 14.0, n_samples)
    rainfall_mm = np.random.uniform(0, 300, n_samples)
    temp_c = np.random.uniform(0, 50, n_samples)
    
    # The Goldilocks Zone: Only survives in these strict ranges
    conditions_met = (
        (ph_level >= 5.5) & (ph_level <= 7.8) &
        (rainfall_mm >= 60) & (rainfall_mm <= 160) &
        (temp_c >= 18) & (temp_c <= 32)
    )
    
    # Add a tiny bit of noise (nature is unpredictable)
    noise = np.random.choice([True, False], n_samples, p=[0.05, 0.95])
    survival = np.where(conditions_met ^ noise, 1, 0) 
    
    agri_data = pd.DataFrame({'ph': ph_level, 'rain': rainfall_mm, 'temp': temp_c, 'survival': survival})
    
    # Random Forest is perfect for non-linear "Goldilocks" boundaries
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf_model.fit(agri_data[['ph', 'rain', 'temp']], agri_data['survival'])
    return rf_model

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
    if val_temp > 32 or val_temp < 18:
        st.caption("- Issue detected: Temperature is outside survivable threshold.")
