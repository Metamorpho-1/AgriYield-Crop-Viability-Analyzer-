# 🌾 AgriYield: Precision Crop Viability Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20GUI-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E.svg)

## Overview
AgriYield is a precision-agriculture classification model. Instead of relying on traditional farming intuition or basic linear estimations, it utilizes a **Random Forest Classifier** pipeline to evaluate the exact probability of a crop's survival based on localized telemetry. 

## 🔬 The Data Architecture & "Goldilocks" Logic
Environmental survival is not linear (e.g., maximum heat does not equal maximum yield). This engine specifically models the "Goldilocks Zone" of agriculture by handling non-monotonic data. It evaluates three primary environmental parameters:
1. **Soil pH Levels** (Optimal threshold: 5.5 - 7.8)
2. **Monthly Rainfall** (Optimal threshold: 60mm - 160mm)
3. **Ambient Temperature** (Optimal threshold: 18°C - 32°C)

By using decision trees, the model correctly penalizes extreme out-of-bounds conditions (like droughts or 50°C heatwaves), outputting a binary classification (`Viable` vs. `High Risk`) alongside a strict probability matrix to help farmers mitigate crop failure.

## 🚀 Quick Start
To run this interactive dashboard locally:

1. Clone the repository
2. Install dependencies: `pip install streamlit pandas numpy scikit-learn`
3. Launch the engine: `streamlit run app.py`

*Built for advanced sustainability data modeling.*
