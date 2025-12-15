# Stroke Risk Prediction System (MLOps Project)

**Kelompok 9 - Kelas RB**

Project ini adalah implementasi End-to-End Machine Learning Operations (MLOps) untuk prediksi risiko stroke, mencakup pipeline training otomatis, komparasi model, dan deployment berbasis Streamlit & Docker.

## 🚀 Fitur Utama

- **Automated Model Comparison**: Membandingkan Logistic Regression, SVM, dan Random Forest secara otomatis.
- **Sensitivity Focused**: Menggunakan metrik _Recall_ untuk meminimalisir False Negative pada diagnosa penyakit.
- **Risk Calibration Layer**: Menggabungkan prediksi AI dengan aturan medis klinis untuk akurasi lebih tinggi.
- **Interactive Dashboard**: Web-based GUI menggunakan Streamlit.

## 🛠️ Tech Stack

- **Language**: Python 3.9
- **Libraries**: Scikit-Learn, Pandas, NumPy, Streamlit
- **Containerization**: Docker

## 📊 Model Performance (Evaluation)

Berdasarkan hasil training terakhir:

- **Random Forest**: Recall 100% (Selected as Best Model) 🏆
- **SVM**: Recall 97.8%
- **Logistic Regression**: Recall 94.8%

## 💻 Cara Menjalankan (Local)

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan training pipeline (Opsional):

   ```bash
   python train_model.py

   ```

3. Jalankan aplikasi:

   ```bash
   streamlit run app.py

   ```

## 🐳 Cara Menjalankan (Docker)

1. Build image:

   ```bash
   docker build -t stroke-prediction .

   ```

2. Run container:

   ```bash
   docker run -p 8501:8501 stroke-prediction

   ```
