import sys
sys.path.append(r"D:\retina_project\venv\Lib\site-packages")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import os

# Hide Tkinter GUI
root = tk.Tk()
root.withdraw()

# Step 1: Train model
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 2: Ask for CSV input file
print("📁 Please select your CSV file for prediction.")
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

if file_path:
    try:
        user_data = pd.read_csv(file_path)
        required_cols = X.columns.tolist()

        if not all(col in user_data.columns for col in required_cols):
            print("❌ Your file must include these columns:")
            print(required_cols)
        else:
            user_input = user_data[required_cols]
            user_data["Prediction"] = model.predict(user_input)

            # Count affected
            affected_count = (user_data["Prediction"] == 1).sum()
            print(f"\n✅ Patients affected by diabetes: {affected_count}\n")

            # Print affected rows
            print("🩺 Affected patient details:")
            print(user_data[user_data["Prediction"] == 1])

            # Save to same file (overwrite original)
            user_data.to_csv(file_path, index=False)
            print(f"\n📌 Predictions added to original file:\n{file_path}")

    except Exception as e:
        print("❌ Error reading the CSV file:", e)
else:
    print("⚠️ No file selected.")
