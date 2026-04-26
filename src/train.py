# ==========================================
# TRAIN FEATURE EXTRACTION - MobileNetV2
# (TANPA FINE-TUNING)
# ==========================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# PATH & PARAMETER
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model_feature_extraction.h5")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "riwayat.pkl")

TRAIN_DIR = os.path.join(BASE_DIR, "dataset_split", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset_split", "val")
TEST_DIR  = os.path.join(BASE_DIR, "dataset_split", "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10   
LEARNING_RATE = 1e-4

# ==========================================
# DATA GENERATOR
# ==========================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass Indices:")
print(train_gen.class_indices)

num_classes = len(train_gen.class_indices)
print(f"Jumlah Kelas: {num_classes}")

# ==========================================
# CLASS WEIGHT
# ==========================================

y_train = train_gen.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(zip(np.unique(y_train), class_weights))

print("\nClass Weight:")
for class_id, weight in class_weights_dict.items():
    class_name = list(train_gen.class_indices.keys())[class_id]
    print(f"{class_name} : {weight:.3f}")

# ==========================================
# BUILD MODEL (FEATURE EXTRACTION)
# ==========================================

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# ❗ TANPA FINE-TUNING → SEMUA DIBEKUKAN
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

callbacks = [checkpoint, earlystop]

# ==========================================
# TRAINING
# ==========================================

print("\nTraining (Feature Extraction Only)")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ==========================================
# EVALUASI TEST SET
# ==========================================

print("\nEvaluasi Test Set")
loss_test, acc_test = model.evaluate(test_gen)

print(f"Akurasi Test Set : {acc_test*100:.2f}%")
print(f"Loss Test Set    : {loss_test:.4f}")

# ==========================================
# SIMPAN MODEL
# ==========================================

model.save(MODEL_PATH)
print(f"\nModel tersimpan di: {MODEL_PATH}")

# ==========================================
# SIMPAN HISTORY
# ==========================================

history_data = {
    "accuracy": history.history['accuracy'],
    "val_accuracy": history.history['val_accuracy'],
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss']
}

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history_data, f)

print(f"History tersimpan di: {HISTORY_PATH}")

# ==========================================
# PLOT HASIL TRAINING
# ==========================================

epochs_range = range(1, len(history_data["accuracy"]) + 1)

plt.figure(figsize=(12,5))

# ========================
# GRAFIK ACCURACY
# ========================
plt.subplot(1,2,1)
plt.plot(epochs_range, history_data["accuracy"])
plt.plot(epochs_range, history_data["val_accuracy"])
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs_range)          
plt.ylim(0, 1)                    
plt.grid()
plt.legend(["Train", "Validation"])

# ========================
# GRAFIK LOSS
# ========================
plt.subplot(1,2,2)
plt.plot(epochs_range, history_data["loss"])
plt.plot(epochs_range, history_data["val_loss"])
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs_range)          
plt.grid()
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()