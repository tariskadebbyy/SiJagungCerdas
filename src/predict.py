# predict.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# ===============================
# SUPPRESS WARNING TF
# ===============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ===============================
# PATH MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_best.h5")

# ===============================
# LOAD MODEL SEKALI SAJA
# ===============================
model = load_model(MODEL_PATH, compile=False)

# ===============================
# LABEL SESUAI TRAINING
# ===============================
LABELS = ["hawar_daun", "karat_daun", "non_jagung", "sehat"]

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ===============================
# PREDIKSI SINGLE IMAGE
# ===============================
def predict_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError("File gambar tidak ditemukan")

    img = preprocess_image(image_path)
    preds = model.predict(img, verbose=0)

    preds = preds[0]

    idx = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    label = LABELS[idx]

    return label, confidence

# ===============================
# TEST MANUAL
# ===============================
if __name__ == "__main__":
    IMAGE_PATH = os.path.join(BASE_DIR, "test_image.jpg")
    label, conf = predict_image(IMAGE_PATH)
    print(f"Hasil Prediksi : {label}")
    print(f"Confidence     : {conf:.2f}%")
