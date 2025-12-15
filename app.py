import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Memuat aset model dan preprocessing yang telah dilatih sebelumnya
model = joblib.load('best_model.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Konfigurasi halaman aplikasi Streamlit
st.set_page_config(page_title="Stroke Risk AI", layout="wide")

# --- SIDEBAR ---
# Menampilkan identitas aplikasi dan informasi tim
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
st.sidebar.title("🏥 MLOps - Kelompok 9")
st.sidebar.subheader("Kelas RB")
st.sidebar.markdown("---")
st.sidebar.write("**Anggota Tim:**")
st.sidebar.text("122450058 - ADIL AULIA RAHMA")
st.sidebar.text("122450061 - Kharisa Harvanny")
st.sidebar.text("122450066 - Cintya Bella")
st.sidebar.text("122450102 - Daris Samudra")
st.sidebar.text("122450059 - NATHANAEL DANIEL")
st.sidebar.markdown("---")
st.sidebar.info(
    "Model ini mengintegrasikan Random Forest Classifier "
    "dengan mekanisme Risk Scoring Calibration untuk meningkatkan interpretabilitas hasil."
)

# --- MAIN CONTENT ---
# Judul utama dan deskripsi singkat aplikasi
st.title("🧠 Stroke Risk Prediction System")
st.write("Silakan masukkan data klinis pasien untuk melakukan prediksi risiko stroke.")

with st.form("patient_data"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profil")
        gender = st.selectbox("Gender", encoders['gender'].classes_)
        age = st.number_input("Umur", 0, 120, 50)
        ever_married = st.selectbox("Status Menikah", encoders['ever_married'].classes_)
        work_type = st.selectbox("Pekerjaan", encoders['work_type'].classes_)

    with col2:
        st.subheader("Medis")
        hypertension = st.radio("Riwayat Hipertensi", ["No", "Yes"])
        heart_disease = st.radio("Sakit Jantung", ["No", "Yes"])
        avg_glucose = st.number_input("Rata-rata Glukosa", 0.0, 300.0, 100.0)

    with col3:
        st.subheader("Gaya Hidup")
        residence = st.selectbox("Tempat Tinggal", encoders['Residence_type'].classes_)
        bmi = st.number_input("BMI", 10.0, 70.0, 28.0)
        smoking = st.selectbox("Status Merokok", encoders['smoking_status'].classes_)

    submit_btn = st.form_submit_button("🔍 ANALISA RISIKO")

if submit_btn:
    # 1. Tahap preprocessing data input pengguna
    # Mengonversi variabel kategorikal biner menjadi numerik
    hyp_val = 1 if hypertension == "Yes" else 0
    heart_val = 1 if heart_disease == "Yes" else 0
    
    # Menyusun data input dalam bentuk DataFrame
    data_input = pd.DataFrame({
        'gender': [encoders['gender'].transform([gender])[0]],
        'age': [age],
        'hypertension': [hyp_val],
        'heart_disease': [heart_val],
        'ever_married': [encoders['ever_married'].transform([ever_married])[0]],
        'work_type': [encoders['work_type'].transform([work_type])[0]],
        'Residence_type': [encoders['Residence_type'].transform([residence])[0]],
        'avg_glucose_level': [avg_glucose],
        'bmi': [bmi],
        'smoking_status': [encoders['smoking_status'].transform([smoking])[0]]
    })
    
    # Normalisasi fitur numerik menggunakan scaler yang telah dilatih
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    data_input[num_cols] = scaler.transform(data_input[num_cols])
    
    # 2. Prediksi probabilitas risiko stroke menggunakan model machine learning
    prob_ai = model.predict_proba(data_input)[0][1]
    
    # 3. Risk Calibration
    # Penyesuaian skor risiko berbasis pengetahuan medis
    # untuk meningkatkan sensitivitas hasil prediksi
    risk_boost = 0.0
    if heart_disease == "Yes": risk_boost += 0.25
    if hypertension == "Yes": risk_boost += 0.20
    if age > 55: risk_boost += 0.15
    if avg_glucose > 200: risk_boost += 0.10
    if smoking in ["smokes", "formerly smoked"]: risk_boost += 0.05
    
    # Menggabungkan probabilitas model dengan skor risiko tambahan
    final_prob = prob_ai + risk_boost
    if final_prob > 0.99:
        final_prob = 0.99  # Membatasi probabilitas maksimum hingga 99%
    
    st.write("---")
    
    # 4. Menampilkan hasil prediksi kepada pengguna
    # Threshold risiko ditetapkan pada 40% untuk meningkatkan sensitivitas deteksi
    if final_prob > 0.40: 
        st.error("⚠️ **PERINGATAN TINGGI!** Pasien Berisiko Stroke.")
        st.metric(
            label="Tingkat Risiko",
            value=f"{final_prob*100:.1f}%",
            delta="BAHAYA",
            delta_color="inverse"
        )
        st.write("**Faktor Risiko Utama:**")
        if heart_disease == "Yes": st.write("- Riwayat Penyakit Jantung")
        if hypertension == "Yes": st.write("- Riwayat Hipertensi")
        if age > 55: st.write("- Faktor Usia Lanjut")
    else:
        st.success("✅ **AMAN.** Risiko Stroke Rendah.")
        st.metric(
            label="Tingkat Risiko",
            value=f"{final_prob*100:.1f}%",
            delta="Normal"
        )
