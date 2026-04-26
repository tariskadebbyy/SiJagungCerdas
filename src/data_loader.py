import os
import pandas as pd

def load_dataset(dataset_dir):
    filepaths, labels = [], []

    for label in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, label)
        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    filepaths.append(os.path.join(class_path, img))
                    labels.append(label)

    return pd.DataFrame({
        'image_path': filepaths,
        'category': labels
    })
