# camera_app.py
import cv2
import numpy as np
import joblib

from preprocess import preprocess_image
from features import extract_features

# Load model
model = joblib.load("model.pkl")

GESTURE_NAMES = {
    "Gesture_0": "Palm",
    "Gesture_1": "Fist",
    "Gesture_2": "Thumbs Up",
    "Gesture_3": "Thumbs Down",
    "Gesture_4": "Index Pointing",
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

cap = cv2.VideoCapture(0)

print("📸 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 🔥 CENTER CROP (CRITICAL)
    size = 300
    cx, cy = w // 2, h // 2
    crop = frame[
        cy - size//2 : cy + size//2,
        cx - size//2 : cx + size//2
    ]

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        features = extract_features(gray)
        features = np.array(features).reshape(1, -1)

        pred = model.predict(features)[0]
        gesture = GESTURE_NAMES.get(pred, pred)

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            frame,
            (cx - size//2, cy - size//2),
            (cx + size//2, cy + size//2),
            (255, 0, 0),
            2
        )

    except:
        pass

    cv2.imshow("Live Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
