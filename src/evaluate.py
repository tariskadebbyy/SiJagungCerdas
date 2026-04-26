# evaluate_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# PATH MODEL & DATA
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.h5")
TEST_DIR = os.path.join(BASE_DIR, "dataset_split", "test")

# ===============================
# EVALUASI MODEL
# ===============================
def evaluate_model(img_size=(224,224), batch_size=16):

    print("Load model:", MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    class_names = list(test_gen.class_indices.keys())

    print("\nPredicting test data...")
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    # ===============================
    # FILTER HAPUS NON_JAGUNG
    # ===============================
    non_jagung_index = class_names.index("non_jagung")

    # 🔥 filter TRUE dan PRED sekaligus
    mask = (y_true != non_jagung_index) & (y_pred != non_jagung_index)

    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # ===============================
    # REMAP LABEL (jadi 0,1,2)
    # ===============================
    unique_labels = sorted(np.unique(y_true_filtered))
    label_mapping = {old: new for new, old in enumerate(unique_labels)}

    y_true_filtered = np.array([label_mapping[y] for y in y_true_filtered])
    y_pred_filtered = np.array([label_mapping[y] for y in y_pred_filtered])

    # ===============================
    # NAMA KELAS BARU (3 kelas)
    # ===============================
    class_names_filtered = [class_names[i] for i in unique_labels]

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(
        y_true_filtered,
        y_pred_filtered,
        labels=list(range(len(class_names_filtered)))
    )

    print("\n=== CONFUSION MATRIX (TANPA NON_JAGUNG) ===")
    print(cm)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        y_true_filtered,
        y_pred_filtered,
        labels=list(range(len(class_names_filtered))),
        target_names=class_names_filtered,
        digits=4
    ))

    # ===============================
    # VISUALISASI CONFUSION MATRIX
    # ===============================
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names_filtered
    )

    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()