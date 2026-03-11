import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load models
# -----------------------------
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")
meta_model = joblib.load("meta_model.pkl")

# Load dataset to get feature names
df = pd.read_csv("breast_cancer.csv")
df = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore')
feature_names = df.columns

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer AI Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Breast Cancer Prediction System")
st.write("AI-powered diagnostic support using an ensemble ML model.")

st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Patient Tumor Measurements")

input_values = {}

for feature in feature_names:
    input_values[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        value=float(df[feature].mean())
    )

input_data = np.array(list(input_values.values())).reshape(1, -1)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("Predict Diagnosis"):

    input_scaled = scaler.transform(input_data)

    # Base model probabilities
    xgb_prob = xgb_model.predict_proba(input_scaled)[:, 1]
    lgbm_prob = lgbm_model.predict_proba(input_scaled)[:, 1]

    meta_features = np.column_stack([xgb_prob, lgbm_prob])

    prediction = meta_model.predict(meta_features)[0]
    probability = meta_model.predict_proba(meta_features)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Malignant (Cancer Detected)")
        st.progress(float(probability))
        st.write(f"Confidence: **{probability*100:.2f}%**")
    else:
        st.success("✅ Benign (No Cancer Detected)")
        st.progress(float(1-probability))
        st.write(f"Confidence: **{(1-probability)*100:.2f}%**")

    st.markdown("---")

    # Show model probabilities
    st.subheader("Model Probabilities")

    prob_df = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "Hybrid Meta Model"],
        "Probability of Malignancy": [
            xgb_prob[0],
            lgbm_prob[0],
            probability
        ]
    })

    st.bar_chart(prob_df.set_index("Model"))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Machine Learning ensemble model (XGBoost + LightGBM + Logistic Regression)")