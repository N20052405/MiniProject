import pickle
import numpy as np
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATA_PATH = 'sign_language_data.pickle' # Renamed for consistency
MODEL_PATH = 'sign_language_model_rf.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pickle'

print("--- STARTING MODEL TRAINING ---")

# 1. DATA LOADING & PREPARATION
try:
    with open(DATA_PATH, 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print(f"FATAL ERROR: {DATA_PATH} not found. Run the data collector first.")
    sys.exit(1)

X_list = []
y_list = []

for sign_label, sequences in data_dict.items():
    for sequence_data in sequences:
        if isinstance(sequence_data, np.ndarray) and sequence_data.size == 126:
            X_list.append(sequence_data)
            y_list.append(sign_label)

try:
    X = np.stack(X_list)
    y = np.array(y_list)
except ValueError as e:
    print("\nFATAL ERROR: Failed to stack data. Check if X_list is empty or shapes mismatch.")
    print(f"Details: {e}")
    sys.exit(1)

# 2. DATA VALIDATION & SPLIT
if X.size == 0 or len(np.unique(y)) < 2:
    print("FATAL ERROR: Not enough valid data or unique classes found.")
    sys.exit(1)

print(f"\nTotal samples loaded: {X.shape[0]}")
print(f"Feature vector size: {X.shape[1]}")
print(f"Total sign classes: {len(np.unique(y))}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded 
)

# 3. MODEL TRAINING
print("\nDefining and training RandomForestClassifier...")
# A max_depth of 10 is often good for initial performance
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. EVALUATION
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Training Complete.")
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# Print a detailed report
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels))


# 5. SAVE ARTIFACTS
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
    
with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model saved as {MODEL_PATH}")
print(f"Label encoder saved as {LABEL_ENCODER_PATH}")