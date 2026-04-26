import os
import shutil
import random

# ===============================
# PATH SESUAI STRUKTUR KAMU
# ===============================
SOURCE_DIR = "../dataset"        
TARGET_DIR = "dataset_split"     

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)

# ===============================
# SPLIT DATASET
# ===============================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end   = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:]
    }

    for split_name, split_images in splits.items():
        target_dir = os.path.join(
            TARGET_DIR, split_name, class_name
        )
        os.makedirs(target_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, img)
            shutil.copy2(src, dst)

    print(f"✅ {class_name}: {total} gambar dipisah")

print("\n🎉 Dataset berhasil dipisahkan ke train / val / test")
