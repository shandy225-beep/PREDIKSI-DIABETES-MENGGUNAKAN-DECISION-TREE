# Upload dataset
from google.colab import files
import pandas as pd


uploaded = files.upload()
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name) # Baca file CSV yang diupload


# Import library yang diperlukan (Split data, DT, Eval, dan Visualisasi)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Pisahkan fitur dan label
X = data.drop(columns='diabetes')
y = data['diabetes']


# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Inisialisasi dan latih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# Prediksi dan evaluasi
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


# Tampilkan hasil evaluasi dan visualisasi confusion matrix
print("Akurasi Model: {:.2f}%".format(accuracy * 100))
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
