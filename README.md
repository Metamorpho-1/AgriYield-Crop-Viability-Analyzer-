# 🌾 AgriYield: Crop Viability Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20GUI-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-F7931E.svg)

## Overview
AgriYield is a precision-agriculture classification model. Instead of relying on traditional farming intuition, it utilizes a Logistic Regression pipeline to evaluate the exact probability of a crop's survival based on localized telemetry. 

## 🔬 The Data Architecture
The engine takes three primary environmental parameters:
1. Soil pH Levels
2. Monthly Rainfall (mm)
3. Average Ambient Temperature (°C)

It outputs a binary classification (`Viable` vs. `High Risk`) alongside a probability matrix to help farmers mitigate crop failure.

## 🚀 Quick Start
To run this interactive dashboard locally:

1. Clone the repository
2. Install dependencies: `pip install streamlit pandas numpy scikit-learn`
3. Launch the engine: `streamlit run app.py`

*Built for advanced sustainability data modeling.*
