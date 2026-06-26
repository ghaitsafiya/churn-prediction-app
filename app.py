import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# =============================================
# LOAD MODEL & ARTIFACT
# =============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_churn.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    with open("top_features.json") as f:
        top_features = json.load(f)
    return model, scaler, label_encoders, top_features

model, scaler, label_encoders, top_features = load_artifacts()

cat_cols_in_features = [c for c in top_features if c in label_encoders]
num_cols_in_features = [c for c in top_features if c not in label_encoders]

# =============================================
# SIDEBAR - INFO
# =============================================
st.sidebar.title("📊 Tentang Aplikasi")
st.sidebar.markdown(
    """
    Aplikasi ini memprediksi apakah seorang pelanggan
    berpotensi **churn** (berhenti berlangganan) atau tidak,
    berdasarkan **15 fitur terpenting** hasil seleksi fitur
    menggunakan Random Forest.

    **Model:** Random Forest (Tuned)  
    **F1-Score:** 0.6672  
    **Recall:** 0.95
    """
)

# =============================================
# HALAMAN UTAMA
# =============================================
st.title("📉 Customer Churn Prediction")
st.markdown("Masukkan data pelanggan untuk memprediksi kemungkinan churn.")
st.markdown("---")

st.subheader("Input Data Pelanggan")

col1, col2, col3 = st.columns(3)
input_data = {}
columns_cycle = [col1, col2, col3]

for i, feature in enumerate(top_features):
    target_col = columns_cycle[i % 3]

    if feature in label_encoders:
        le = label_encoders[feature]
        options = list(le.classes_)
        input_data[feature] = target_col.selectbox(
            f"{feature}", options, key=feature
        )
    else:
        if feature in ["satisfaction_score", "nps_score"]:
            input_data[feature] = target_col.slider(
                f"{feature}", 0.0, 10.0, 5.0, key=feature
            )
        elif feature in ["delivery_delay_days"]:
            input_data[feature] = target_col.number_input(
                f"{feature}", min_value=0, max_value=30, value=3, key=feature
            )
        else:
            input_data[feature] = target_col.number_input(
                f"{feature}", min_value=0.0, value=0.0, format="%.2f", key=feature
            )

st.markdown("---")

if st.button("🔍 Prediksi Churn", type="primary"):
    df_input = pd.DataFrame([input_data])[top_features]

    for col in cat_cols_in_features:
        le = label_encoders[col]
        df_input[col] = le.transform(df_input[col])

    df_input_scaled = scaler.transform(df_input)

    pred = model.predict(df_input_scaled)[0]
    pred_proba = model.predict_proba(df_input_scaled)[0]

    st.markdown("### Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠️ Pelanggan **berpotensi CHURN** "
                 f"(probabilitas: {pred_proba[1]*100:.1f}%)")
    else:
        st.success(f"✅ Pelanggan **tidak berpotensi churn** "
                   f"(probabilitas: {pred_proba[0]*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh(["Tidak Churn", "Churn"], pred_proba,
            color=["#2ecc71", "#e74c3c"])
    ax.set_xlim(0, 1)
    for i, v in enumerate(pred_proba):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center")
    ax.set_title("Probabilitas Prediksi")
    st.pyplot(fig)

st.markdown("---")
st.caption("Dibuat sebagai proyek UAS Mata Kuliah Bengkel Koding — "
           "Ghaitsa Qiyala Shafiya (A11.2023.14976)")
