"""
========================================================================
DECISION TREE CLASSIFIER - DIABETES DETECTION
Kelompok 3 - Diabetes Detection Project
========================================================================
Deskripsi:
Script ini mengimplementasikan model Decision Tree Classifier untuk
memprediksi diabetes berdasarkan fitur-fitur yang telah dipilih melalui
Mutual Information feature selection.

Proses:
1. Upload dan load dataset
2. Split data menjadi training dan testing set (80:20)
3. Training model Decision Tree
4. Evaluasi model dengan confusion matrix dan accuracy
5. Visualisasi hasil evaluasi
========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os


# ========================================================================
# KONFIGURASI
# ========================================================================
DATASET_FILE = 'reduced_6Fitur_diabetes_dataset.csv'
TEST_SIZE = 0.2  # 20% untuk testing, 80% untuk training
RANDOM_STATE = 42


# ========================================================================
# FUNGSI UTILITY
# ========================================================================
def load_dataset(file_path):
    """Load dataset dari file CSV"""
    print(f"[INFO] Loading dataset dari: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset berhasil dimuat")
        print(f"[INFO] Shape: {data.shape}")
        print(f"[INFO] Kolom: {list(data.columns)}")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan: {file_path}")
        print("[HINT] Pastikan file berada di direktori yang sama atau berikan path lengkap")
        return None
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return None


def prepare_data(data, target_column='diabetes'):
    """Pisahkan fitur dan label"""
    print("\n[INFO] Memisahkan fitur dan label...")
    if target_column not in data.columns:
        print(f"[ERROR] Kolom '{target_column}' tidak ditemukan!")
        return None, None
    
    X = data.drop(columns=target_column)
    y = data[target_column]
    
    print(f"[SUCCESS] Fitur (X): {X.shape}")
    print(f"[SUCCESS] Label (y): {y.shape}")
    print(f"[INFO] Distribusi kelas: {dict(y.value_counts())}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data menjadi training dan testing set"""
    print(f"\n[INFO] Splitting data (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Menjaga proporsi kelas
    )
    
    print(f"[SUCCESS] Training set: {X_train.shape[0]} samples")
    print(f"[SUCCESS] Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, random_state=42):
    """Inisialisasi dan latih model Decision Tree"""
    print("\n[INFO] Training Decision Tree model...")
    model = DecisionTreeClassifier(
        random_state=random_state,
        criterion='gini',  # atau 'entropy'
        max_depth=None,     # Bisa diatur untuk mencegah overfitting
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    model.fit(X_train, y_train)
    print("[SUCCESS] Model berhasil dilatih")
    print(f"[INFO] Jumlah node: {model.tree_.node_count}")
    print(f"[INFO] Kedalaman tree: {model.tree_.max_depth}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluasi model dan return metrik"""
    print("\n[INFO] Evaluasi model...")
    y_pred = model.predict(X_test)
    
    # Hitung metrik evaluasi
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return {
        'predictions': y_pred,
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def display_results(results):
    """Tampilkan hasil evaluasi"""
    print("\n" + "="*70)
    print("HASIL EVALUASI MODEL")
    print("="*70)
    print(f"Akurasi      : {results['accuracy']*100:.2f}%")
    print(f"Precision    : {results['precision']*100:.2f}%")
    print(f"Recall       : {results['recall']*100:.2f}%")
    print(f"F1-Score     : {results['f1_score']*100:.2f}%")
    print("="*70)


def plot_confusion_matrix(conf_matrix, save_fig=False, filename='confusion_matrix.png'):
    """Visualisasi confusion matrix"""
    print("\n[INFO] Membuat visualisasi confusion matrix...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix - Decision Tree Classifier', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Confusion matrix disimpan: {filename}")
    
    plt.show()
    print("[SUCCESS] Visualisasi selesai")


def plot_feature_importance(model, feature_names, save_fig=False, filename='feature_importance.png'):
    """Visualisasi feature importance"""
    print("\n[INFO] Membuat visualisasi feature importance...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance - Decision Tree', fontsize=14, fontweight='bold', pad=20)
    plt.bar(range(len(importances)), importances[indices], color='skyblue', edgecolor='navy')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Feature importance disimpan: {filename}")
    
    plt.show()
    print("[SUCCESS] Visualisasi selesai")


def plot_decision_tree(model, feature_names, class_names, save_fig=False, filename='decision_tree.png'):
    """Visualisasi struktur decision tree"""
    print("\n[INFO] Membuat visualisasi decision tree...")
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Decision tree disimpan: {filename}")
    
    plt.show()
    print("[SUCCESS] Visualisasi selesai")


# ========================================================================
# MAIN PROGRAM
# ========================================================================
def main():
    print("\n" + "="*70)
    print("DECISION TREE CLASSIFIER - DIABETES DETECTION")
    print("Kelompok 3 - Diabetes Detection Project")
    print("="*70)
    
    # 1. Load dataset
    data = load_dataset(DATASET_FILE)
    if data is None:
        print("\n[ERROR] Proses dihentikan karena dataset tidak dapat dimuat")
        return
    
    # 2. Prepare data
    X, y = prepare_data(data)
    if X is None or y is None:
        return
    
    # 3. Split data
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    
    # 4. Train model
    model = train_model(X_train, y_train, RANDOM_STATE)
    
    # 5. Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # 6. Display results
    display_results(results)
    
    # 7. Visualizations
    plot_confusion_matrix(results['confusion_matrix'], save_fig=True)
    plot_feature_importance(model, X.columns, save_fig=True)
    
    # Optional: Plot decision tree (might be large)
    # plot_decision_tree(model, X.columns, model.classes_, save_fig=True)
    
    # 8. Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, results['predictions']))
    print("="*70)
    
    print("\n[COMPLETED] Proses selesai!")
    print("[INFO] Semua visualisasi telah disimpan sebagai file PNG")


# ========================================================================
# UNTUK GOOGLE COLAB
# ========================================================================
def run_in_colab():
    """
    Fungsi khusus untuk menjalankan di Google Colab dengan file upload
    """
    print("="*70)
    print("RUNNING IN GOOGLE COLAB MODE")
    print("="*70)
    
    # Upload dataset
    from google.colab import files
    
    print("\n[INFO] Silakan upload file dataset...")
    uploaded = files.upload()
    
    if not uploaded:
        print("[ERROR] Tidak ada file yang diupload!")
        return
    
    file_name = list(uploaded.keys())[0]
    print(f"[SUCCESS] File uploaded: {file_name}")
    
    # Load dan proses data
    data = pd.read_csv(file_name)
    print(f"[INFO] Dataset shape: {data.shape}")
    
    # Lanjutkan dengan proses yang sama
    X, y = prepare_data(data)
    if X is None or y is None:
        return
    
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    model = train_model(X_train, y_train, RANDOM_STATE)
    results = evaluate_model(model, X_test, y_test)
    
    display_results(results)
    plot_confusion_matrix(results['confusion_matrix'])
    plot_feature_importance(model, X.columns)
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, results['predictions']))


if __name__ == "__main__":
    # Deteksi environment
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        run_in_colab()
    else:
        main()
