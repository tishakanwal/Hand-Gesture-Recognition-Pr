# train.py
import os
import numpy as np
import joblib

from preprocess import preprocess_image
from features import extract_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


ROOT_DATASET = "hg14/HG14-Hand Gesture"

X, y, class_names = [], [], []

print("[INFO] Loading dataset...")

for label, folder in enumerate(sorted(os.listdir(ROOT_DATASET))):
    folder_path = os.path.join(ROOT_DATASET, folder)

    if not os.path.isdir(folder_path):
        continue

    class_names.append(folder)

    for file in os.listdir(folder_path):
        if file.startswith("."):
            continue

        try:
            img = preprocess_image(os.path.join(folder_path, file))
            feat = extract_features(img)

            X.append(feat)
            y.append(label)

        except:
            continue

X = np.array(X)
y = np.array(y)

print("[INFO] Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("[INFO] Training optimized Random Forest...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=120)),   # 🔥 best balance
    ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))
print("======================================")

joblib.dump(model, "model.pkl")

print("[INFO] Model saved successfully")