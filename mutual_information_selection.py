"""
========================================================================
MUTUAL INFORMATION FEATURE SELECTION
Kelompok 3 - Diabetes Detection Project
========================================================================
Deskripsi:
Script ini melakukan feature selection menggunakan metode Mutual Information
untuk mengurangi dimensi dataset diabetes dan memilih fitur-fitur yang
paling informatif untuk prediksi diabetes.

Proses:
1. Load dataset diabetes yang sudah dibersihkan
2. Encode fitur kategorikal menggunakan LabelEncoder
3. Hitung Mutual Information score untuk setiap fitur
4. Filter fitur berdasarkan threshold
5. Simpan dataset baru dengan fitur terpilih
========================================================================
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import os


# ========================================================================
# KONFIGURASI
# ========================================================================
INPUT_FILE = r"C:\Users\rifki\Downloads\cleaned_diabetes_dataset.csv"
OUTPUT_FILE = 'reduced_6Fitur_diabetes_dataset.csv'
MI_THRESHOLD = 0.01  # Threshold untuk seleksi fitur
RANDOM_STATE = 42


# ========================================================================
# FUNGSI UTAMA
# ========================================================================
def load_dataset(file_path):
    """Load dataset dari file CSV"""
    print(f"[INFO] Loading dataset dari: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset berhasil dimuat. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return None


def encode_categorical_features(df):
    """Encode fitur kategorikal menjadi numerik"""
    print("\n[INFO] Encoding fitur kategorikal...")
    df_encoded = df.copy()
    label_encoders = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    print(f"[INFO] Fitur kategorikal yang ditemukan: {list(categorical_columns)}")
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        print(f"  - {col}: {len(le.classes_)} unique values")
    
    print("[SUCCESS] Encoding selesai")
    return df_encoded, label_encoders


def calculate_mutual_information(X, y, random_state=42):
    """Hitung Mutual Information score untuk setiap fitur"""
    print("\n[INFO] Menghitung Mutual Information scores...")
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=random_state)
    
    # Buat DataFrame dengan hasil MI
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI Score': mi_scores
    })
    mi_df = mi_df.sort_values(by='MI Score', ascending=False)
    
    print("[SUCCESS] Perhitungan MI selesai")
    return mi_df


def select_features(mi_df, threshold):
    """Pilih fitur berdasarkan threshold MI score"""
    print(f"\n[INFO] Memilih fitur dengan MI Score >= {threshold}")
    selected_features = mi_df[mi_df['MI Score'] >= threshold]['Feature'].tolist()
    print(f"[SUCCESS] {len(selected_features)} fitur terpilih")
    return selected_features


def save_reduced_dataset(df, selected_features, target_column, output_file):
    """Simpan dataset dengan fitur terpilih"""
    print(f"\n[INFO] Menyimpan dataset baru ke: {output_file}")
    new_df = df[selected_features + [target_column]]
    new_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Dataset disimpan. Shape: {new_df.shape}")
    print(f"[INFO] Fitur terpilih: {selected_features}")
    return new_df


def display_mi_results(mi_df):
    """Tampilkan hasil Mutual Information"""
    print("\n" + "="*70)
    print("NILAI MUTUAL INFORMATION UNTUK SETIAP FITUR")
    print("="*70)
    print(mi_df.to_string(index=False))
    print("="*70)


# ========================================================================
# MAIN PROGRAM
# ========================================================================
def main():
    print("\n" + "="*70)
    print("MUTUAL INFORMATION FEATURE SELECTION")
    print("Diabetes Detection Project - Kelompok 3")
    print("="*70)
    
    # 1. Load dataset
    df = load_dataset(INPUT_FILE)
    if df is None:
        return
    
    # 2. Encode fitur kategorikal
    df_encoded, label_encoders = encode_categorical_features(df)
    
    # 3. Pisahkan fitur dan target
    if 'diabetes' not in df_encoded.columns:
        print("[ERROR] Kolom 'diabetes' tidak ditemukan dalam dataset!")
        return
    
    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']
    print(f"\n[INFO] Jumlah fitur: {X.shape[1]}")
    print(f"[INFO] Jumlah sampel: {X.shape[0]}")
    
    # 4. Hitung mutual information
    mi_df = calculate_mutual_information(X, y, RANDOM_STATE)
    
    # 5. Tampilkan hasil
    display_mi_results(mi_df)
    
    # 6. Pilih fitur dengan threshold
    selected_features = select_features(mi_df, MI_THRESHOLD)
    
    # 7. Simpan dataset baru
    new_df = save_reduced_dataset(df, selected_features, 'diabetes', OUTPUT_FILE)
    
    print("\n[COMPLETED] Proses feature selection selesai!")
    print(f"[INFO] Dataset asli: {df.shape[1]} fitur")
    print(f"[INFO] Dataset baru: {len(selected_features)} fitur")
    print(f"[INFO] Reduksi: {df.shape[1] - len(selected_features)} fitur")


if __name__ == "__main__":
    main()
