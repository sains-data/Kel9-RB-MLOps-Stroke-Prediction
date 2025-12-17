import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# Library evaluasi model untuk kebutuhan laporan
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("=== MEMULAI PROSES TRAINING DAN EVALUASI MODEL ===")

# 1. PEMUATAN DATASET
# Dataset prediksi stroke dimuat dari file CSV
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Menghapus kolom ID karena tidak berpengaruh terhadap prediksi
df = df.drop('id', axis=1)

# Menghapus data dengan kategori gender 'Other' karena jumlahnya sangat kecil
df = df[df['gender'] != 'Other']

# 2. TAHAP PREPROCESSING DATA

# Menangani missing value pada kolom BMI menggunakan nilai rata-rata
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# Encoding fitur kategorikal menggunakan Label Encoding
encoders = {}
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. PENANGANAN KETIDAKSEIMBANGAN DATA (OVERSAMPLING)
# Oversampling dilakukan untuk meningkatkan sensitivitas model
# terhadap kasus pasien yang mengalami stroke
print("Melakukan oversampling pada data pasien stroke...")
stroke_data = df[df['stroke'] == 1]

# Data stroke diperbanyak untuk mengurangi bias kelas mayoritas
df = pd.concat([df] + [stroke_data] * 50, axis=0)

# 4. NORMALISASI FITUR NUMERIK
# Standardisasi dilakukan agar fitur numerik memiliki skala yang sama
scaler = StandardScaler()
num_cols = ['age', 'avg_glucose_level', 'bmi']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. PEMBAGIAN DATA TRAINING DAN TESTING
X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. TRAINING DAN EVALUASI MODEL
# Evaluasi difokuskan pada akurasi dan recall untuk mendeteksi risiko stroke
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\nHASIL EVALUASI MODEL BERDASARKAN CONFUSION MATRIX")
print("=" * 60)

for name, model in models.items():
    print(f"\nMODEL: {name}")
    
    # Melatih model menggunakan data training
    model.fit(X_train, y_train)
    
    # Melakukan prediksi pada data testing
    y_pred = model.predict(X_test)
    
    # Perhitungan metrik evaluasi
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # Menampilkan hasil evaluasi
    print(f"   Akurasi  : {acc*100:.2f}%")
    print(f"   Recall   : {recall*100:.2f}% (Sensitivitas Deteksi Stroke)")
    print(f"   Confusion Matrix:")
    print(cm)
    print("\n   Classification Report:")
    print(report)
    print("-" * 60)
    
    # Menyimpan model ke dalam file
    filename = f"model_{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"   Model berhasil disimpan sebagai: {filename}")

# Menyimpan encoder dan scaler untuk proses deployment
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n=== PROSES TRAINING DAN EVALUASI SELESAI ===")
print("Model, encoder, dan scaler siap digunakan untuk tahap deployment.")
