# Prediksi Diabetes Menggunakan Decision Tree

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan sistem prediksi diabetes menggunakan algoritma **Decision Tree Classifier** dengan **Mutual Information** untuk feature selection. Proyek ini dirancang untuk dijalankan di **Google Colab**.

### Fitur Utama

- ✅ **Feature Selection** menggunakan Mutual Information
- ✅ **Decision Tree Classifier** untuk prediksi diabetes
- ✅ **Evaluasi Model** dengan Accuracy dan Confusion Matrix
- ✅ **Visualisasi** Confusion Matrix menggunakan Seaborn
- ✅ **Optimized** untuk Google Colab

---

## 📁 Struktur Proyek

```
PREDIKSI-DIABETES-MENGGUNAKAN-DECISION-TREE/
│
├── mutual_information_selection.py        # Script untuk feature selection
├── decision_tree_model.py                 # Script untuk training dan evaluasi model
├── diabetes_prediction_dataset.csv.zip    # Dataset diabetes (compressed)
├── .gitignore                             # File yang diabaikan Git
└── README.md                              # Dokumentasi proyek
```

---

## 🚀 Cara Menggunakan

### Metode:  Google Colab (Recommended)

#### 1. **Persiapan Dataset**

1. Extract file `diabetes_prediction_dataset.csv. zip`
2. Bersihkan dataset jika diperlukan dan simpan sebagai `cleaned_diabetes_dataset.csv`

#### 2. **Feature Selection dengan Mutual Information**

Upload dan jalankan `mutual_information_selection.py` di Google Colab:

```python
# Upload file mutual_information_selection.py ke Colab
# Kemudian jalankan: 
%run mutual_information_selection.py
```

Atau copy-paste kode langsung ke cell Colab dan jalankan.

**Output:**
- Tabel Mutual Information scores untuk setiap fitur
- File `reduced_6Fitur_diabetes_dataset.csv` dengan fitur terpilih

#### 3. **Training dan Evaluasi Model**

Upload dan jalankan `decision_tree_model.py` di Google Colab:

```python
# Upload file decision_tree_model.py ke Colab
# Kemudian jalankan:
%run decision_tree_model.py
```

**Proses yang terjadi:**
1. Upload dataset hasil feature selection (`reduced_6Fitur_diabetes_dataset.csv`)
2. Split data menjadi training (80%) dan testing (20%)
3. Training model Decision Tree
4. Evaluasi model dan tampilkan akurasi
5. Visualisasi Confusion Matrix

**Output:**
- Akurasi Model (dalam persentase)
- Confusion Matrix (visualisasi)

---

## 📊 Detail Implementasi

### 1. Mutual Information Feature Selection

**File:** `mutual_information_selection.py`

**Fungsi:**
- Load dataset diabetes yang sudah dibersihkan
- Encode fitur kategorikal menggunakan LabelEncoder
- Hitung Mutual Information score untuk setiap fitur
- Filter fitur dengan MI Score >= threshold (default:  0.01)
- Simpan dataset baru dengan fitur terpilih

**Konfigurasi:**
```python
INPUT_FILE = r"C:\Users\rifki\Downloads\cleaned_diabetes_dataset. csv"
OUTPUT_FILE = 'reduced_6Fitur_diabetes_dataset.csv'
MI_THRESHOLD = 0.01
RANDOM_STATE = 42
```

**⚠️ Catatan:** Ubah `INPUT_FILE` sesuai dengan lokasi dataset Anda, atau upload file melalui Google Colab. 

**Alur Kerja:**
1. Load dataset CSV
2. Encode semua fitur kategorikal ke numerik
3. Pisahkan fitur (X) dan target (y)
4. Hitung MI scores untuk setiap fitur
5.  Tampilkan dan urutkan berdasarkan MI Score
6. Filter fitur dengan score >= threshold
7. Simpan dataset baru dengan fitur terpilih

---

### 2. Decision Tree Classifier

**File:** `decision_tree_model.py`

**Fungsi:**
- Upload dataset hasil feature selection melalui Colab
- Split data menjadi training (80%) dan testing (20%)
- Training model Decision Tree
- Prediksi dan evaluasi model
- Visualisasi Confusion Matrix

**Konfigurasi:**
```python
TEST_SIZE = 0.2        # 20% untuk testing
RANDOM_STATE = 42      # Untuk reproducibility
```

**Parameter Model:**
```python
DecisionTreeClassifier(random_state=42)
```

**Alur Kerja:**
1. Upload dataset melalui `files.upload()`
2. Baca CSV menggunakan pandas
3. Pisahkan fitur (X) dan label (y)
4. Split data dengan `train_test_split`
5. Training model Decision Tree
6. Prediksi pada data testing
7. Hitung accuracy dan confusion matrix
8. Tampilkan hasil dan visualisasi

---

## 📈 Metrik Evaluasi

Model dievaluasi menggunakan metrik berikut:

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | Persentase prediksi yang benar dari seluruh data testing |
| **Confusion Matrix** | Matrix yang menunjukkan True Positive, True Negative, False Positive, dan False Negative |

**Interpretasi Confusion Matrix:**

```
                Predicted
                No    Yes
Actual  No     TN     FP
        Yes    FN     TP
```

- **True Negative (TN):** Prediksi No, Aktual No ✅
- **False Positive (FP):** Prediksi Yes, Aktual No ❌
- **False Negative (FN):** Prediksi No, Aktual Yes ❌
- **True Positive (TP):** Prediksi Yes, Aktual Yes ✅

---

## 🔧 Kustomisasi

### Mengubah Threshold Mutual Information

Edit di `mutual_information_selection.py`:

```python
threshold = 0.05  # Ubah nilai threshold (default: 0.01)
```

- Nilai threshold **lebih tinggi** = fitur yang dipilih **lebih sedikit** (lebih selektif)
- Nilai threshold **lebih rendah** = fitur yang dipilih **lebih banyak**

### Mengubah Parameter Decision Tree

Edit di `decision_tree_model.py`:

```python
model = DecisionTreeClassifier(
    random_state=42,
    criterion='entropy',      # Gunakan 'gini' atau 'entropy'
    max_depth=10,             # Batasi kedalaman tree
    min_samples_split=10,     # Minimal sampel untuk split node
    min_samples_leaf=5        # Minimal sampel di leaf node
)
```

**Parameter penting:**
- `criterion`: Metode split ('gini' atau 'entropy')
- `max_depth`: Kedalaman maksimal tree (None = unlimited)
- `min_samples_split`: Minimal sampel untuk split node
- `min_samples_leaf`: Minimal sampel di setiap leaf

### Mengubah Rasio Train/Test Split

Edit di `decision_tree_model.py`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # Ubah menjadi 70% train, 30% test
)
```

---

## 📊 Dataset

### Format Dataset

Dataset harus berformat CSV dengan struktur: 

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| gender | Categorical | Jenis kelamin |
| age | Numeric | Usia pasien |
| hypertension | Numeric | Status hipertensi (0/1) |
| heart_disease | Numeric | Status penyakit jantung (0/1) |
| smoking_history | Categorical | Riwayat merokok |
| bmi | Numeric | Body Mass Index |
| HbA1c_level | Numeric | Level HbA1c |
| blood_glucose_level | Numeric | Level glukosa darah |
| diabetes | Categorical | Target variable (0/1) |

### Lokasi Dataset

- **Input:** `cleaned_diabetes_dataset.csv` (dataset yang sudah dibersihkan)
- **Output:** `reduced_6Fitur_diabetes_dataset. csv` (hasil feature selection)
- **Archived:** `diabetes_prediction_dataset.csv. zip` (dataset asli compressed)

**⚠️ Catatan:** Dataset CSV diabaikan oleh Git (lihat `.gitignore`), jadi Anda perlu extract dan prepare dataset sendiri.

---

## 📦 Dependencies

Proyek ini membutuhkan library Python berikut:

```python
pandas                 # Data manipulation
numpy                  # Numerical computing
scikit-learn           # Machine learning (DecisionTreeClassifier, train_test_split, metrics)
matplotlib             # Plotting
seaborn                # Statistical visualization
```

**Install di Google Colab:**

Semua library sudah ter-install secara default di Google Colab. Jika ada yang kurang, install dengan:

```bash
!pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🎯 Panduan Lengkap Google Colab

### Step-by-Step: 

**1. Buka Google Colab**
- Kunjungi [colab.research.google.com](https://colab.research.google.com)
- Buat notebook baru

**2. Feature Selection**

Cell 1:
```python
# Upload mutual_information_selection.py
from google.colab import files
uploaded = files.upload()
```

Cell 2:
```python
# Jalankan feature selection
%run mutual_information_selection.py
```

Cell 3:
```python
# Download hasil feature selection
files.download('reduced_6Fitur_diabetes_dataset.csv')
```

**3. Training Model**

Cell 1:
```python
# Upload decision_tree_model.py
from google.colab import files
uploaded = files.upload()
```

Cell 2:
```python
# Jalankan training model
%run decision_tree_model.py
# Akan muncul prompt untuk upload dataset hasil feature selection
```

**Output yang diharapkan:**
```
Akurasi Model: XX. XX%
[Visualisasi Confusion Matrix]
```

---

## 💡 Tips & Best Practices

### 1. **Handling Path Issues di Colab**

Jika `mutual_information_selection.py` error karena path, ubah baris 6 menjadi:

```python
# Dari:
df = pd.read_csv(r"C:\Users\rifki\Downloads\cleaned_diabetes_dataset. csv")

# Menjadi (upload file):
from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)
```

### 2. **Evaluasi Lebih Detail**

Tambahkan metrik evaluasi lain di `decision_tree_model.py`:

```python
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Setelah y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
```

### 3. **Cross-Validation**

Untuk evaluasi lebih robust:

```python
from sklearn.model_selection import cross_val_score

# Sebelum train_test_split, tambahkan:
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
```

### 4. **Feature Importance**

Lihat fitur mana yang paling penting dalam prediksi:

```python
# Setelah model. fit()
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualisasi
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

### 5. **Mencegah Overfitting**

Modifikasi parameter model:

```python
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,              # Batasi kedalaman
    min_samples_split=20,     # Minimal 20 sampel untuk split
    min_samples_leaf=10       # Minimal 10 sampel di leaf
)
```

---

## 🐛 Troubleshooting

### Error: "File not found" di mutual_information_selection.py

**Solusi:**
```python
# Gunakan upload file Colab
from google.colab import files
uploaded = files. upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)
```

### Error: "KeyError: 'diabetes'"

**Solusi:**
- Pastikan kolom target bernama 'diabetes'
- Cek nama kolom:  `print(df.columns)`
- Ubah nama kolom jika perlu:  `df.rename(columns={'old_name':  'diabetes'}, inplace=True)`

### Warning: Mixed types in LabelEncoder

**Solusi:**
```python
# Ubah tipe data ke string dulu
for col in df_encoded.select_dtypes(include=['object']).columns:
    df_encoded[col] = df_encoded[col].astype(str)
    le = LabelEncoder()
    df_encoded[col] = le. fit_transform(df_encoded[col])
```

### Model Accuracy Rendah

**Penyebab dan Solusi:**
1. **Data tidak seimbang** → Gunakan SMOTE atau class_weight
2. **Overfitting** → Kurangi max_depth, tingkatkan min_samples_split
3. **Underfitting** → Tambah max_depth, coba Random Forest
4. **Feature kurang informatif** → Turunkan MI threshold atau feature engineering

---

## 📚 Resources

### Dokumentasi

- [scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Mutual Information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Google Colab Guide](https://colab.research.google.com/notebooks/welcome. ipynb)

### Dataset Source

- Dataset diabetes prediction dari Kaggle atau sumber lainnya
- Format:  CSV dengan kolom fitur kesehatan dan target diabetes

---

## 👥 Kelompok 3

**Diabetes Prediction Project**

Developed for Machine Learning course 🎓

---

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ Initial release
- ✅ Mutual Information feature selection implementation
- ✅ Decision Tree Classifier implementation
- ✅ Basic evaluation metrics (Accuracy, Confusion Matrix)
- ✅ Visualization with Seaborn
- ✅ Google Colab optimized

---

## 📄 License

This project is created for educational purposes. 

---

## 🆘 Support

Jika ada pertanyaan atau masalah: 
- Buat issue di repository ini
- Hubungi tim pengembang Kelompok 3

**Happy Coding!  🚀📊🤖**
