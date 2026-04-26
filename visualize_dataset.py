import os
import matplotlib.pyplot as plt

# ===============================
# PATH DATASET
# ===============================
image_dir = "dataset_split/train"  # lebih tepat pakai train

# Ambil daftar kelas (folder)
class_folders = sorted([
    f for f in os.listdir(image_dir)
    if os.path.isdir(os.path.join(image_dir, f))
])

# ===============================
# TAMPILKAN CONTOH GAMBAR
# ===============================
for class_name in class_folders:
    class_path = os.path.join(image_dir, class_name)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(images) == 0:
        continue

    num_images = min(5, len(images))

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(f"Contoh Gambar Kelas: {class_name}", fontsize=14)

    # Kalau cuma 1 gambar, axes bukan array
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        img_path = os.path.join(class_path, images[i])
        img = plt.imread(img_path)

        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
