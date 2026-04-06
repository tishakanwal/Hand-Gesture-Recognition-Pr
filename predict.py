# predict.py
import numpy as np
import joblib

from preprocess import preprocess_image
from features import extract_features

model = joblib.load("model.pkl")

GESTURE_NAMES = {
    "Gesture_0": "Palm",
    "Gesture_1": "Fist",
    "Gesture_2": "Thumbs Up",
    "Gesture_3": "Thumbs Down",
    "Gesture_4": "Pointing",
    "Gesture_5": "Victory",
    "Gesture_6": "OK",
    "Gesture_7": "Stop",
    "Gesture_8": "Call Me",
    "Gesture_9": "Rock",
    "Gesture_10": "Like",
    "Gesture_11": "Dislike",
    "Gesture_12": "Peace",
    "Gesture_13": "Closed Hand"
}

def predict_gesture(image_path):
    img = preprocess_image(image_path)
    features = extract_features(img)

    features = np.array(features).reshape(1, -1)

    pred = model.predict(features)[0]

    return GESTURE_NAMES.get(pred, pred)