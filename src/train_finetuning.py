# ==========================================
# TRAIN FINETUNING - MobileNetV2
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

MODEL_PATH = os.path.join(MODEL_DIR, "model_final_mobilenetv2.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.pkl")

TRAIN_DIR = os.path.join(BASE_DIR, "dataset_split", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset_split", "val")
TEST_DIR  = os.path.join(BASE_DIR, "dataset_split", "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5
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

print("\n📌 Class Indices:")
print(train_gen.class_indices)

num_classes = len(train_gen.class_indices)
print(f"📌 Jumlah Kelas: {num_classes}")

# ==========================================
# CLASS WEIGHT (untuk data imbalance)
# ==========================================

y_train = train_gen.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(zip(np.unique(y_train), class_weights))

print("\n📌 Class Weight:")
for class_id, weight in class_weights_dict.items():
    class_name = list(train_gen.class_indices.keys())[class_id]
    print(f"{class_name} : {weight:.3f}")

# ==========================================
# BUILD MODEL
# ==========================================

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Stage 1: Freeze base model
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
# TRAINING STAGE 1
# ==========================================

print("\n📌 Stage 1: Training classifier (base frozen)")
history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ==========================================
# TRAINING STAGE 2 (Fine-tuning)
# ==========================================

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n📌 Stage 2: Fine-tuning top layers")
history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ==========================================
# EVALUASI TEST SET
# ==========================================

print("\n📌 Evaluasi Test Set")
loss_test, acc_test = model.evaluate(test_gen)

print(f"🎯 Akurasi Test Set : {acc_test*100:.2f}%")
print(f"📉 Loss Test Set    : {loss_test:.4f}")

# ==========================================
# SIMPAN MODEL FINAL
# ==========================================

model.save(MODEL_PATH)
print(f"\n✅ Model akhir tersimpan di: {MODEL_PATH}")

# ==========================================
# SIMPAN HISTORY UNTUK JUPYTER
# ==========================================

acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
loss = history_stage1.history['loss'] + history_stage2.history['loss']
val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

history_combined = {
    "accuracy": acc,
    "val_accuracy": val_acc,
    "loss": loss,
    "val_loss": val_loss
}

with open(HISTORY_PATH, "wb") as f:
    pickle.dump(history_combined, f)

print(f"📊 History tersimpan di: {HISTORY_PATH}")

# ==========================================
# PLOT TRAINING RESULT
# ==========================================

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc)
plt.plot(epochs_range, val_acc)
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Val"])

plt.subplot(1,2,2)
plt.plot(epochs_range, loss)
plt.plot(epochs_range, val_loss)
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Val"])

plt.tight_layout()
plt.show()
