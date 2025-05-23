title: "Practical Machine Learning: Prediction Assignment Writeup"
author: "Shintia Lestari Putri"
date: "`r Sys.Date()`"
output: html_document

#Introduction
Membangun model pembelajaran mesin untuk memprediksi tipe latihanberdasarkan data sensor dari wearable
Pelatihan dataset berisi berbagai fitur dari sensor gerakan seperti:

roll_beltpitch_beltBahasa Indonesia :yaw_belt

total_accel_belt, gyros_arm_x, `akseleratoraccel_arm_y, dll

Target: classe,menunjukkan metode latihan yang dilakukan (A, B, C, D, E).

---
# 1. Import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load data
train_data = pd.read_csv("training.csv")  # ganti dengan path file Anda
test_data = pd.read_csv("testing.csv")    # ganti dengan path file Anda

# 3. Pembersihan data
# Hapus kolom dengan banyak missing value dan non-numeric
threshold = 0.9 * train_data.shape[0]
train_data = train_data.dropna(thresh=threshold, axis=1)
test_data = test_data[train_data.columns.drop('classe')]

# Hapus kolom yang tidak relevan
irrelevant_cols = ['X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2',
                   'cvtd_timestamp', 'new_window', 'num_window']
train_data = train_data.drop(columns=irrelevant_cols, errors='ignore')
test_data = test_data.drop(columns=irrelevant_cols, errors='ignore')

# 4. Split data
X = train_data.drop('classe', axis=1)
y = train_data['classe']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Standardisasi (optional untuk RF, tapi penting jika pakai model lain)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# 6. Model: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# 7. Evaluasi Model
y_pred = rf.predict(X_valid_scaled)
print("Akurasi Validasi:", accuracy_score(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred))

# Visualisasi Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8. Cross-validation
cv_scores = cross_val_score(rf, scaler.transform(X), y, cv=5)
print("Cross-Validation Accuracy Mean:", np.mean(cv_scores))

# 9. Prediksi pada data uji
test_scaled = scaler.transform(test_data)
test_predictions = rf.predict(test_scaled)

# 10. Output hasil prediksi
print("Prediksi 20 Kasus Uji:\n", test_predictions)


*Report submitted by Shintia Lestari Putri on `r Sys.Date()`*
