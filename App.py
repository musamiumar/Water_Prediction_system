# Import necessary libraries
import tkinter as tk
from tkinter import *
import random
import customtkinter as ctk  # Assuming this is a custom module you've created
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from tkinter import messagebox
import numpy as np
from sklearn.utils import resample
from sklearn.utils import shuffle

# Load and preprocess data
# Assuming 'water_potability.csv' contains your data
df = pd.read_csv("water_potability.csv")
df = df.dropna()  # Drop rows with missing values
notp = df[df['Potability'] == 0]
p = df[df['Potability'] == 1]
oversampled = resample(p, replace=True, n_samples=1200)  # Oversample the minority class

df = pd.concat([notp, oversampled])
df = shuffle(df)
x = df.drop(['Potability'], axis=1)
y = df['Potability']
scaler = StandardScaler()
columns = x.columns
x[columns] = scaler.fit_transform(x[columns])
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

# Define models
models = {
    "LR": LogisticRegression(C=0.1, penalty='l2'),
    "SVM": SVC(C=10, degree=2, kernel='rbf'),
    "DT": DecisionTreeClassifier(criterion='gini', max_depth=38, min_samples_leaf=1),
    "RF": RandomForestClassifier(min_samples_leaf=2, n_estimators=500)
}

# Train models
for model_name, model in models.items():
    model.fit(X_train, Y_train)

# Create GUI
ctk.set_default_color_theme("blue")

main_frame = tk.Tk()
main_frame.title("Water Prediction System")
main_frame.attributes("-fullscreen", True)
main_frame.configure(background="#2C3E50")

# Create GUI elements
ctk.CTkLabel(main_frame, text="WATER PREDICTION SYSTEM", width=10, font=("Roboto", 15, "bold")).pack(pady=0)
entries = []  # List to store entry widgets
entry_ranges = [(0, 14), (47, 323), (320, 6123), (0, 14), (129, 482), (181, 753), (2, 29), (0, 125), (1, 7)]  # Ranges for random number generation

# Function to return information for display
def return_info(acc, cm, model):
    return f" \n {model}: \n Accuracy: {acc} \n Confusion Matrix: \n {cm} \n"

# Function to generate random numbers for entries
def generate_random_numbers():
    for entry, (min_val, max_val) in zip(entries, entry_ranges):
        entry.delete(0, END)
        entry.insert(0, str(random.randint(min_val, max_val)))

display = ""  # Variable for displaying output
selected_model = IntVar()  # Variable to store selected model

# Function to classify and display results
def classify():
    if selected_model.get() == 0:
        messagebox.showerror("Error", "Please select a model.")
        return

    values = scaler.transform([[float(entry.get()) for entry in entries]])
    model_name = list(models.keys())[selected_model.get() - 1]
    prediction = models[model_name].predict(values)[0]
    prediction_result = "Water is potable 👍🏼" if prediction == 1 else "Not potable 🤮"
    result_label.configure(text=f"Predicted result: {prediction_result}")

    acc = accuracy_score(Y_test, models[model_name].predict(X_test))
    cm = confusion_matrix(Y_test, models[model_name].predict(X_test))
    classification_rep = classification_report(Y_test, models[model_name].predict(X_test))

    output_label.configure(text=f"Results: \n {model_name}: \n Accuracy: {acc * 100}% \n Confusion Matrix: \n {cm} \n")

    messagebox.showinfo("Classification Report", classification_rep)

# Create labels and entry widgets
labels = ["PH (0-14)", "HARDNESS (47, 323)", "SOLIDS (320, 6123)", "CHLOROMINES (0, 14)", "SULFATE (129, 482)", "CONDUCTIVITY (181, 753)", "ORGANIC CARBON (2, 29)", "TRICHLOROMETHANE (0, 125)", "TURBIDITY (1, 7)"]
for label, (min_val, max_val) in zip(labels, entry_ranges):
    ctk.CTkLabel(main_frame, text=label, width=10, corner_radius=5, font=("Roboto", 10, "bold")).pack(pady=2)
    entry = ctk.CTkEntry(main_frame, width=100, corner_radius=5)
    entry.pack()
    entries.append(entry)

# Create radio buttons for model selection
for i, model_name in enumerate(models.keys()):
    ctk.CTkRadioButton(main_frame, text=model_name, variable=selected_model, value=i + 1, font=("Roboto", 10)).pack(pady=3, padx=0)

# Create buttons for generating random numbers and classifying
generate_button = ctk.CTkButton(main_frame, text="Generate Random Numbers", corner_radius=10, height=20, fg_color="#E74C3C", hover_color="#C0392B", font=("Roboto", 10, "bold"), command=generate_random_numbers)
generate_button.pack(pady=2)

classify_button = ctk.CTkButton(main_frame, text="Classify", font=("Roboto", 10, "bold"), corner_radius=10, width=30, height=20, fg_color="#E74C3C", hover_color="#C0392B", command=classify)
classify_button.pack(pady=2)

# Labels for displaying results
result_label = ctk.CTkLabel(main_frame, text="", width=20, corner_radius=5, font=("Roboto", 10, "bold"))
result_label.pack(pady=5)

output_label = ctk.CTkLabel(main_frame, text=display, font=('Roboto', 15, 'bold'))
output_label.pack(pady=5)

main_frame.mainloop()
