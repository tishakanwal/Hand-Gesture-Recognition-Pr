# preprocess.py
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not readable")

    # 🔥 THIS SIZE WORKED BEST EARLIER
    img = cv2.resize(img, (64, 64))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray