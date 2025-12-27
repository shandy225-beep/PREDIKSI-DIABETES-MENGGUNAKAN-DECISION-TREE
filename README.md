# Diabetes Detection using Decision Tree Classifier

##  Deskripsi Proyek

Proyek ini mengimplementasikan sistem deteksi diabetes menggunakan algoritma **Decision Tree Classifier** dengan **Mutual Information** untuk feature selection. Proyek ini dikembangkan oleh **Kelompok 3** sebagai bagian dari pembelajaran Machine Learning.

### Fitur Utama

-  **Feature Selection** menggunakan Mutual Information
-  **Decision Tree Classifier** untuk prediksi diabetes
-  **Evaluasi Model** lengkap (Accuracy, Precision, Recall, F1-Score)
-  **Visualisasi** Confusion Matrix dan Feature Importance
-  **Support** untuk local environment dan Google Colab

---

##  Struktur Proyek

```
DT_Diabetes/

 mutual_information_selection.py   # Script untuk feature selection
 decision_tree_model.py            # Script untuk training dan evaluasi model
 requirements.txt                   # Daftar dependencies
 README.md                          # Dokumentasi proyek
 .gitignore                         # File yang diabaikan Git

 cleaned_diabetes_dataset.csv       # Dataset asli (tidak di-commit)
 reduced_6Fitur_diabetes_dataset.csv # Dataset hasil feature selection
```

---

##  Cara Menggunakan

### Metode 1: Local Environment (Python)

#### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

#### 2. **Feature Selection dengan Mutual Information**

```bash
python mutual_information_selection.py
```

**Output:**
- Tabel Mutual Information scores untuk setiap fitur
- File educed_6Fitur_diabetes_dataset.csv dengan fitur terpilih

#### 3. **Training dan Evaluasi Model**

```bash
python decision_tree_model.py
```

**Output:**
- Metrik evaluasi (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix (saved as PNG)
- Feature importance chart (saved as PNG)
- Classification report

---

### Metode 2: Google Colab

#### 1. **Upload File ke Colab**

Upload kedua file Python ke Google Colab:
- mutual_information_selection.py
- decision_tree_model.py

#### 2. **Jalankan Feature Selection**

```python
# Jalankan di cell pertama
!python mutual_information_selection.py
```

#### 3. **Jalankan Model Decision Tree**

```python
# Jalankan di cell kedua
!python decision_tree_model.py
```

Atau gunakan fungsi un_in_colab() yang sudah disediakan:

```python
from decision_tree_model import run_in_colab
run_in_colab()
```

---

##  Detail Implementasi

### 1. Mutual Information Feature Selection

**File:** mutual_information_selection.py

**Proses:**
1. Load dataset diabetes yang sudah dibersihkan
2. Encode fitur kategorikal menggunakan LabelEncoder
3. Hitung Mutual Information score untuk setiap fitur
4. Filter fitur dengan MI Score >= threshold (default: 0.01)
5. Simpan dataset baru dengan fitur terpilih

**Konfigurasi:**
```python
INPUT_FILE = r"C:\Users\rifki\Downloads\cleaned_diabetes_dataset.csv"
OUTPUT_FILE = 'reduced_6Fitur_diabetes_dataset.csv'
MI_THRESHOLD = 0.01
RANDOM_STATE = 42
```

**Fungsi Utama:**
- load_dataset() - Load dataset dari CSV
- encode_categorical_features() - Encode fitur kategorikal
- calculate_mutual_information() - Hitung MI scores
- select_features() - Pilih fitur berdasarkan threshold
- save_reduced_dataset() - Simpan dataset baru

---

### 2. Decision Tree Classifier

**File:** decision_tree_model.py

**Proses:**
1. Load dataset hasil feature selection
2. Split data menjadi training (80%) dan testing (20%)
3. Training model Decision Tree
4. Evaluasi model dengan berbagai metrik
5. Visualisasi hasil evaluasi

**Konfigurasi:**
```python
DATASET_FILE = 'reduced_6Fitur_diabetes_dataset.csv'
TEST_SIZE = 0.2  # 20% untuk testing
RANDOM_STATE = 42
```

**Parameter Model:**
```python
DecisionTreeClassifier(
    random_state=42,
    criterion='gini',      # Metrik split: 'gini' atau 'entropy'
    max_depth=None,        # Kedalaman maksimal tree
    min_samples_split=2,   # Minimum sampel untuk split
    min_samples_leaf=1     # Minimum sampel di leaf node
)
```

**Fungsi Utama:**
- load_dataset() - Load dataset
- prepare_data() - Pisahkan fitur dan label
- split_data() - Split train/test dengan stratification
- 	rain_model() - Training Decision Tree
- evaluate_model() - Evaluasi dengan multiple metrics
- plot_confusion_matrix() - Visualisasi confusion matrix
- plot_feature_importance() - Visualisasi feature importance
- plot_decision_tree() - Visualisasi struktur tree

---

##  Metrik Evaluasi

Model dievaluasi menggunakan metrik berikut:

| Metrik | Deskripsi |
|--------|-----------|
| **Accuracy** | Persentase prediksi yang benar |
| **Precision** | Ketepatan prediksi positif |
| **Recall** | Kemampuan mendeteksi kasus positif |
| **F1-Score** | Harmonic mean dari Precision dan Recall |
| **Confusion Matrix** | Matrix prediksi vs aktual |

---

##  Kustomisasi

### Mengubah Threshold Mutual Information

Edit di mutual_information_selection.py:

```python
MI_THRESHOLD = 0.05  # Ubah nilai threshold (default: 0.01)
```

Nilai threshold lebih tinggi = fitur yang dipilih lebih sedikit (lebih selektif)

### Mengubah Parameter Decision Tree

Edit di decision_tree_model.py:

```python
model = DecisionTreeClassifier(
    criterion='entropy',     # Gunakan entropy sebagai ganticriterion
    max_depth=10,            # Batasi kedalaman untuk mencegah overfitting
    min_samples_split=10,    # Minimal sampel untuk split node
    min_samples_leaf=5       # Minimal sampel di leaf node
)
```

### Mengubah Rasio Train/Test Split

Edit di decision_tree_model.py:

```python
TEST_SIZE = 0.3  # Ubah menjadi 70% train, 30% test
```

---

##  Dataset

### Format Dataset

Dataset harus berformat CSV dengan struktur:

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| Feature 1 | Numeric/Categorical | Fitur prediksi |
| Feature 2 | Numeric/Categorical | Fitur prediksi |
| ... | ... | ... |
| diabetes | Categorical | Target variable (0/1 atau No/Yes) |

### Lokasi Dataset

- **Input:** cleaned_diabetes_dataset.csv (dataset asli)
- **Output:** educed_6Fitur_diabetes_dataset.csv (hasil feature selection)

**Catatan:** Ubah path di konfigurasi sesuai lokasi dataset Anda.

---

##  Visualisasi

### 1. Confusion Matrix

![Confusion Matrix Example](confusion_matrix.png)

Menampilkan:
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

### 2. Feature Importance

![Feature Importance Example](feature_importance.png)

Menampilkan kontribusi setiap fitur dalam prediksi model.

### 3. Decision Tree Structure (Optional)

Visualisasi struktur lengkap Decision Tree (dapat menghasilkan file besar).

---

##  Dependencies

Proyek ini membutuhkan library Python berikut:

```
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
scikit-learn>=1.2.0    # Machine learning
matplotlib>=3.6.0      # Plotting
seaborn>=0.12.0        # Statistical visualization
```

Install semua dependencies:

```bash
pip install -r requirements.txt
```

---

##  Tips & Best Practices

### 1. **Handling Imbalanced Data**

Jika dataset tidak seimbang, gunakan:
```python
from sklearn.utils import resample
# Atau
from imblearn.over_sampling import SMOTE
```

### 2. **Cross-Validation**

Untuk evaluasi lebih robust:
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

### 3. **Hyperparameter Tuning**

Gunakan GridSearchCV:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### 4. **Mencegah Overfitting**

- Batasi max_depth
- Tingkatkan min_samples_split dan min_samples_leaf
- Gunakan pruning
- Lakukan cross-validation

---

##  Troubleshooting

### Error: "File not found"

**Solusi:**
- Pastikan path dataset sudah benar
- Gunakan absolute path atau pastikan file ada di direktori yang sama

```python
INPUT_FILE = r"C:\path\to\your\cleaned_diabetes_dataset.csv"
```

### Error: "Column 'diabetes' not found"

**Solusi:**
- Periksa nama kolom target di dataset
- Ubah 	arget_column di fungsi prepare_data()

```python
X, y = prepare_data(data, target_column='your_target_column_name')
```

### Warning: "LabelEncoder with mixed types"

**Solusi:**
- Pastikan fitur kategorikal sudah dalam tipe data yang konsisten
- Convert ke string terlebih dahulu jika perlu

```python
df[col] = df[col].astype(str)
```

### Model Accuracy rendah

**Solusi:**
1. Periksa distribusi kelas (imbalanced?)
2. Lakukan hyperparameter tuning
3. Coba algoritma lain (Random Forest, XGBoost)
4. Tambah atau ubah fitur (feature engineering)
5. Periksa data quality (missing values, outliers)

---

##  Resources

### Dokumentasi

- [scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Mutual Information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Referensi

- Breiman, L. (1984). Classification and Regression Trees
- Shannon, C. E. (1948). A Mathematical Theory of Communication

---

##  Kelompok 3

**Diabetes Detection Project**

Developed with  for Machine Learning course

---

##  License

This project is created for educational purposes.

---

##  Changelog

### Version 1.0.0
-  Initial release
-  Mutual Information feature selection
-  Decision Tree Classifier implementation
-  Comprehensive evaluation metrics
-  Visualization tools
-  Google Colab support

---

##  Support

Jika ada pertanyaan atau masalah, silakan buat issue di repository ini atau hubungi tim pengembang.

**Happy Coding! **
