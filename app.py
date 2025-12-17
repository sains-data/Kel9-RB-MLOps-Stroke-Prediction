import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Memuat encoder dan scaler hasil training
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Konfigurasi halaman aplikasi
st.set_page_config(page_title="Stroke Risk AI", layout="wide")

# ===================== SIDEBAR =====================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3004/3004458.png",
    width=100
)
st.sidebar.title("MLOps - Kelompok 9")
st.sidebar.subheader("Kelas RB")

# --------- Pemilihan Model ---------
st.sidebar.markdown("---")
st.sidebar.write("**Konfigurasi Model**")

model_choice = st.sidebar.selectbox(
    "Pilih Algoritma Machine Learning:",
    ["Random Forest", "SVM", "Logistic Regression"]
)

st.sidebar.info(f"Model yang digunakan: **{model_choice}**")

# --------- Informasi Tim ---------
st.sidebar.markdown("---")
st.sidebar.write("**Anggota Tim:**")
st.sidebar.text("122450058 - Adil Aulia Rahma")
st.sidebar.text("122450061 - Kharisa Harvanny")
st.sidebar.text("122450066 - Cintya Bella")
st.sidebar.text("122450102 - Daris Samudra")
st.sidebar.text("122450059 - Nathanael Daniel")

# ===================== LOAD MODEL =====================
# Model dimuat berdasarkan pilihan pengguna
if model_choice == "Random Forest":
    model = joblib.load('model_random_forest.pkl')
elif model_choice == "SVM":
    model = joblib.load('model_svm.pkl')
else:
    model = joblib.load('model_logistic_regression.pkl')

# ===================== MAIN CONTENT =====================
st.title("Stroke Risk Prediction System")
st.write(
    "Aplikasi ini digunakan untuk memprediksi risiko stroke "
    "berdasarkan data klinis dan gaya hidup pasien."
)

# ------------------ Form Input Pasien ------------------
with st.form("patient_data"):
    col1, col2, col3 = st.columns(3)

    # ---- Data Profil ----
    with col1:
        st.subheader("Profil Pasien")
        gender = st.selectbox("Jenis Kelamin", encoders['gender'].classes_)
        age = st.number_input(
            "Usia (tahun)", min_value=1, max_value=120, value=50
        )
        ever_married = st.selectbox(
            "Status Pernikahan", encoders['ever_married'].classes_
        )
        work_type = st.selectbox(
            "Jenis Pekerjaan", encoders['work_type'].classes_
        )

    # ---- Data Medis ----
    with col2:
        st.subheader("Riwayat Medis")
        hypertension = st.radio("Hipertensi", ["No", "Yes"])
        heart_disease = st.radio("Penyakit Jantung", ["No", "Yes"])
        avg_glucose = st.number_input(
            "Rata-rata Kadar Glukosa",
            min_value=50.0,
            max_value=300.0,
            value=100.0
        )

    # ---- Gaya Hidup ----
    with col3:
        st.subheader("Gaya Hidup")
        residence = st.selectbox(
            "Tipe Tempat Tinggal", encoders['Residence_type'].classes_
        )
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 70.0, 28.0)
        smoking = st.selectbox(
            "Status Merokok", encoders['smoking_status'].classes_
        )

    submit_btn = st.form_submit_button("Analisis Risiko")

# ===================== PROSES PREDIKSI =====================
if submit_btn:
    # Konversi input biner ke numerik
    hyp_val = 1 if hypertension == "Yes" else 0
    heart_val = 1 if heart_disease == "Yes" else 0

    # Membentuk DataFrame sesuai format data training
    data_input = pd.DataFrame({
        'gender': [encoders['gender'].transform([gender])[0]],
        'age': [age],
        'hypertension': [hyp_val],
        'heart_disease': [heart_val],
        'ever_married': [encoders['ever_married'].transform([ever_married])[0]],
        'work_type': [encoders['work_type'].transform([work_type])[0]],
        'Residence_type': [
            encoders['Residence_type'].transform([residence])[0]
        ],
        'avg_glucose_level': [avg_glucose],
        'bmi': [bmi],
        'smoking_status': [
            encoders['smoking_status'].transform([smoking])[0]
        ]
    })

    # Standardisasi fitur numerik
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    data_input[num_cols] = scaler.transform(data_input[num_cols])

    # Prediksi probabilitas risiko stroke menggunakan model
    prob_ai = model.predict_proba(data_input)[0][1]

    # Penyesuaian risiko berbasis faktor klinis
    risk_boost = 0.0
    if heart_disease == "Yes": 
        risk_boost += 0.25
    if hypertension == "Yes": 
        risk_boost += 0.20
    if age > 55: 
        risk_boost += 0.15
    if avg_glucose > 200: 
        risk_boost += 0.10
    if smoking in ["smokes", "formerly smoked"]: 
        risk_boost += 0.05

    final_prob = prob_ai + risk_boost
    if final_prob > 0.99:
        final_prob = 0.99

    st.write("---")

    # ===================== OUTPUT HASIL =====================
    if final_prob > 0.40:
        st.error(
            "Risiko stroke pasien tergolong **tinggi**. "
            "Disarankan untuk melakukan pemeriksaan medis lanjutan."
        )
        st.metric(
            label=f"Tingkat Risiko ({model_choice})",
            value=f"{final_prob*100:.1f}%",
            delta="Tinggi",
            delta_color="inverse"
        )
        st.write("**Faktor Risiko Dominan:**")
        if heart_disease == "Yes": 
            st.write("- Riwayat penyakit jantung")
        if hypertension == "Yes": 
            st.write("- Riwayat hipertensi")
        if age > 55: 
            st.write("- Usia lanjut")
    else:
        st.success(
            "Risiko stroke pasien tergolong **rendah** "
            "berdasarkan data yang dimasukkan."
        )
        st.metric(
            label=f"Tingkat Risiko ({model_choice})",
            value=f"{final_prob*100:.1f}%",
            delta="Normal"
        )
