import sys
sys.path.append(r"D:\retina_project\venv\Lib\site-packages")

# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

# Load CSV from user selection
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select your CSV file")
if not file_path:
    print("No file selected.")
    exit()

# Load dataset
data = pd.read_csv(file_path)
print("Columns in dataset:", list(data.columns))

# Show missing values
print("\nMissing values in each column:\n", data.isnull().sum())

# Suggest binary columns
binary_candidates = [col for col in data.columns if data[col].nunique() == 2]
print("\nSuggested binary target columns:", binary_candidates)

# Get target column
target_col = input("Enter the name of the target column (e.g., creditworthy): ").strip()
if target_col not in data.columns:
    print("Invalid target column.")
    exit()

X = data.drop(columns=[target_col])
y = data[target_col]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ROC AUC
if len(np.unique(y_test)) < 2:
    print("\nROC AUC Score: Not defined (only one class present in test set).")
else:
    auc = roc_auc_score(y_test, y_prob)
    print("\nROC-AUC Score:", auc)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()
