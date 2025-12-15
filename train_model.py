import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

print("Memulai proses pelatihan model untuk memperoleh sensitivitas (recall) terbaik.")

# 1. LOAD DATA
# Membaca dataset kesehatan dan melakukan pembersihan awal
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.drop('id', axis=1)
df = df[df['gender'] != 'Other']

# 2. PREPROCESSING
# Menangani missing value pada fitur BMI menggunakan mean imputation
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# Encoding variabel kategorikal menggunakan Label Encoder
encoders = {}
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- STRATEGI OVERSAMPLING ---
# Melakukan penyeimbangan kelas dengan memperbanyak data pasien stroke
print("Melakukan proses oversampling pada data pasien stroke.")
stroke_data = df[df['stroke'] == 1]

# Menggabungkan data asli dengan hasil duplikasi data stroke
df = pd.concat([df] + [stroke_data] * 50, axis=0) 

# 3. SCALE DATA
# Normalisasi fitur numerik menggunakan Standard Scaler
scaler = StandardScaler()
num_cols = ['age', 'avg_glucose_level', 'bmi']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. SPLIT DATA
# Membagi data menjadi data latih dan data uji
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. KOMPARASI MODEL
# Membandingkan tiga algoritma klasifikasi berdasarkan nilai recall
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

best_model = None
best_recall = 0  # Fokus utama evaluasi adalah recall (sensitivitas)
best_name = ""

print("\nHasil perbandingan model dengan fokus pada nilai recall (sensitivitas):")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluasi performa model
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} -> Recall Stroke: {recall*100:.2f}% | Akurasi: {acc*100:.2f}%")
    
    # Menentukan model terbaik berdasarkan nilai recall tertinggi
    if recall > best_recall:
        best_recall = recall
        best_model = model
        best_name = name

print("-" * 60)
print(f"Model terbaik berdasarkan evaluasi recall adalah: {best_name}")

# 6. SIMPAN MODEL DAN ARTEFAK
# Menyimpan model terbaik, encoder, dan scaler untuk digunakan pada tahap deployment
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nProses penyimpanan model dan artefak pendukung telah selesai.")
