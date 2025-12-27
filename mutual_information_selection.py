import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv(r"C:\Users\rifki\Downloads\cleaned_diabetes_dataset.csv")

# 2. Encode fitur kategorikal
df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# 3. Pisahkan fitur dan target
X = df_encoded.drop('diabetes', axis=1)
y = df_encoded['diabetes']

# 4. Hitung mutual information
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# 5. Tampilkan skor mutual information
print("Nilai Mutual Information untuk setiap fitur:")
print(mi_df)

# 6. Tentukan ambang batas (threshold)
threshold = 0.01 

# 7. Pilih fitur dengan MI Score >= threshold
selected_features = mi_df[mi_df['MI Score'] >= threshold]['Feature'].tolist()

# 8. Buat dataset baru dengan fitur terpilih dan target
new_df = df[selected_features + ['diabetes']]

# 9. Simpan dataset baru ke file
new_df.to_csv('reduced_6Fitur_diabetes_dataset.csv', index=False)
print(f"\nDataset baru disimpan dengan {len(selected_features)} fitur + target.")

