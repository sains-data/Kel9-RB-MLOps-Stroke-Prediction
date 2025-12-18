# 🚑 Sistem Prediksi Risiko Stroke  
**Proyek Machine Learning Operations (MLOps)**

---

## 📌 Gambaran Umum
Proyek ini merupakan implementasi **end-to-end Machine Learning Operations (MLOps)** untuk membangun sistem **prediksi risiko stroke** berbasis data kesehatan pasien.  
Sistem dirancang tidak hanya untuk menghasilkan model machine learning yang akurat, tetapi juga memastikan **reproducibility, modularitas, dan kesiapan deploy**.

Pendekatan yang digunakan menggabungkan:
- Pipeline machine learning terstruktur
- Penanganan data medis secara khusus
- Aplikasi web interaktif untuk inference
- Containerization menggunakan Docker  

Sehingga sistem dapat digunakan sebagai **alat bantu skrining risiko stroke** yang praktis dan aman.

---

## 🧠 Latar Belakang & Permasalahan

Stroke merupakan salah satu penyebab utama kematian dan kecacatan jangka panjang.  
Dalam konteks skrining kesehatan, **false negative** (pasien berisiko tetapi diprediksi aman) jauh lebih berbahaya dibandingkan **false positive**, karena dapat menyebabkan keterlambatan penanganan medis.

Oleh karena itu, proyek ini berfokus pada pembangunan sistem prediksi risiko stroke dengan **sensitivitas (recall) yang tinggi**, yang lebih mengutamakan **deteksi dini** dibandingkan sekadar akurasi numerik, dengan menerapkan prinsip **Machine Learning Operations (MLOps)**.
---
## 🎯 Tujuan Proyek

- Membangun model prediksi risiko stroke dengan fokus pada **sensitivitas (recall)**  
- Menerapkan prinsip **MLOps** untuk memastikan pipeline yang **reproducible dan terkelola dengan baik**  
- Menyediakan aplikasi prediksi risiko stroke yang **siap digunakan secara lokal**  
---

## 📊 Dataset
- **Nama Dataset**: Healthcare Stroke Dataset  
- **Format**: CSV (Tabular)  
- **Variabel Target**: `stroke`
  - `0` = Tidak berisiko stroke  
  - `1` = Berisiko stroke  

### Fitur Utama
- Umur (age)  
- Hipertensi (hypertension)  
- Penyakit jantung (heart_disease)  
- Rata-rata kadar glukosa (avg_glucose_level)  
- Body Mass Index (BMI)  
- Jenis kelamin, status merokok, tipe tempat tinggal, dan fitur pendukung lainnya  

### Tantangan Data
- **Missing value** pada kolom BMI  
- **Fitur kategorikal** berupa data teks  
- **Ketidakseimbangan data (imbalanced data)** antara pasien stroke dan non-stroke  

---

## ⚖️ Penanganan Imbalanced Data (Tahap Kritis)
Dataset bersifat **sangat tidak seimbang**, di mana jumlah pasien stroke jauh lebih sedikit dibandingkan pasien sehat.  
Jika model dilatih tanpa penanganan khusus, model akan bias dan cenderung memprediksi semua pasien sebagai tidak stroke.

### Solusi: Oversampling
- Data pasien stroke diduplikasi hingga **50 kali**
- Bertujuan memaksa model mempelajari pola stroke secara lebih mendalam
- Strategi ini digunakan untuk menghasilkan sistem **high sensitivity screening**

> Dalam konteks medis, sistem yang terlalu waspada lebih aman dibandingkan sistem yang melewatkan risiko penyakit serius.

---

## 🔁 Pipeline Sistem (Pseudo-MLOps Pipeline)

Pipeline pada proyek ini dirancang mengikuti prinsip **Machine Learning Operations (MLOps)**, dengan pemisahan yang jelas antara proses *training*, *artifact management*, dan *inference*.  
Meskipun belum menggunakan orkestrator otomatis, seluruh alur dibuat **reproducible, modular, dan siap dikembangkan** ke sistem MLOps yang lebih kompleks.

---

### 1️⃣ Training Pipeline (Offline Process)
Training pipeline diimplementasikan pada file `train_model.py`.

Tahapan yang dilakukan:
1. **Data Ingestion**  
   Dataset dimuat dari file CSV sebagai sumber data utama.

2. **Data Validation & Cleaning**  
   - Kolom ID dihapus karena tidak relevan  
   - Missing value pada BMI ditangani menggunakan imputasi rata-rata  

3. **Feature Engineering**  
   - Fitur kategorikal dikonversi menjadi numerik menggunakan encoding  
   - Fitur numerik distandarisasi menggunakan scaler  

4. **Handling Imbalanced Data**  
   Oversampling agresif dilakukan pada data pasien stroke untuk mengatasi ketidakseimbangan kelas.

5. **Model Training & Comparison**  
   Beberapa algoritma (Logistic Regression, SVM, dan Random Forest) dilatih dan dibandingkan menggunakan dataset yang sama.

6. **Model Evaluation**  
   Evaluasi difokuskan pada metrik **Recall (Sensitivitas)** untuk meminimalkan risiko false negative.

7. **Artifact Generation**  
   Model terlatih serta artefak preprocessing (encoder dan scaler) disimpan dalam format `.pkl`.

---

### 2️⃣ Artifact Management
Artefak hasil training disimpan terpisah dari kode aplikasi, meliputi:
- Model machine learning terlatih  
- Encoder fitur kategorikal  
- Scaler fitur numerik  

Pendekatan ini memastikan:
- Konsistensi antara training dan inference  
- Tidak diperlukan pelatihan ulang saat aplikasi dijalankan  
- Sistem bersifat reproducible  

---

### 3️⃣ Inference Pipeline (Online Process)
Inference pipeline diimplementasikan pada file `app.py`.

Alur inference:
1. Model dan artefak preprocessing dimuat saat aplikasi dijalankan  
2. Pengguna memasukkan data pasien melalui aplikasi web  
3. Data input diproses menggunakan encoder dan scaler yang sama dengan training  
4. Model menghasilkan probabilitas risiko stroke  
5. Sistem menerapkan **risk calibration** berbasis pengetahuan medis  
6. Hasil prediksi ditampilkan secara visual kepada pengguna  

---

### 4️⃣ Deployment & Reproducibility
Seluruh sistem dikemas menggunakan **Docker** untuk memastikan:
- Konsistensi lingkungan eksekusi  
- Kemudahan deployment lintas sistem operasi  
- Menghindari masalah *“it works on my machine”*  

---

### Diagram Pseudo-MLOps Pipeline
```mermaid
flowchart LR
    A[Data Source] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Model Registry]
    G --> H[Deployment]
    H --> I[Monitoring]
    I --> B










