# ==========================================
# K-Nearest Neighbors (KNN) Classification
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==========================================
# Load Dataset
# ==========================================

current_folder = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_folder, "Iris.csv")

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!\n")

print(df.head())

# ==========================================
# Features and Target
# ==========================================

X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# Train KNN Model
# ==========================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================
# Prediction
# ==========================================

y_pred = knn.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")

print(cm)

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# ==========================================
# Heatmap
# ==========================================

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    cmap="Blues",

    fmt="d"

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ==========================================
# Accuracy for Different K Values
# ==========================================

accuracy_list = []

for k in range(1,21):

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy_list.append(accuracy_score(y_test, pred))

plt.figure(figsize=(8,5))

plt.plot(range(1,21), accuracy_list, marker="o")

plt.title("Accuracy vs K")

plt.xlabel("K Value")

plt.ylabel("Accuracy")

plt.grid()

plt.show()

print("\nTask Completed Successfully!")
