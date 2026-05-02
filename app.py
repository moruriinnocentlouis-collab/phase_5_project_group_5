import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load model
model = joblib.load("stacking_ensemble.pkl")

# feature order
feature_order = [
    'Total Assets',
    'Total Deposit liabilities',
    'Total Equity',
    'Total Liabilities',
    'Cash & Cash Equivalent',
    'Financial Investments',
    'Gross Loan Portfolio',
    'Net Loan Portfolio',
    'Share Capital',
    'Statutory Reserve'
]

# page config
st.set_page_config(page_title="SACCO Predictor", layout="wide")

st.title("🏦 SACCO Surplus Prediction App")

st.image("sacco_image.jpg", use_container_width=True)

st.markdown(
"""
This tool predicts SACCO surplus based on key financial indicators.

👉 Enter realistic financial values (in KES) to get accurate predictions.
"""
)

# Load sample data
try:
    df_sample = pd.read_csv("sample_data.csv")

    st.markdown("## 📊 Key Drivers of SACCO Performance")

    st.markdown(
        """
        **Insight:**  
        Total Liabilities and Total Equity are the strongest predictors of surplus across all ensemble models.  
        This suggests that capital structure plays a more significant role than lending volume in determining SACCO performance.
        """
    )

except:
    pass

# layout in two columns
col1, col2 = st.columns(2)

inputs = {}

# split features across columns
for i, feature in enumerate(feature_order):
    if i < len(feature_order)//2:
        with col1:
            inputs[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=1000000.0,
                step=100000.0,
                format="%.2f"
            )
    else:
        with col2:
            inputs[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=1000000.0,
                step=100000.0,
                format="%.2f"
            )

# prediction button
st.markdown("---")

if st.button("📊 Predict Surplus"):

    input_df = pd.DataFrame([inputs])[feature_order]

    # apply same transformation
    input_log = np.log1p(input_df)

    prediction_log = model.predict(input_log)
    prediction = np.expm1(prediction_log)

    st.success(f"💰 Predicted Surplus: KES {prediction[0]:,.2f}")