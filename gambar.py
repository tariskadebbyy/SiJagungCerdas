import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# === Nama kelas ===
classes = ['Hawar', 'Karat', 'Sehat']

# === Path dataset (PASTIKAN SESUAI FOLDER KAMU) ===
base_dir = r'C:\Corn_Detection\dataset'

paths = [
    os.path.join(base_dir, 'hawar_daun'),
    os.path.join(base_dir, 'karat_daun'),
    os.path.join(base_dir, 'sehat')
]

# === Cek apakah folder ada ===
for path in paths:
    if not os.path.exists(path):
        print(f"Folder tidak ditemukan: {path}")
        exit()

plt.figure(figsize=(9, 9))

# === Loop tiap kelas ===
for i, path in enumerate(paths):
    images = os.listdir(path)

    # Filter hanya file gambar
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0:
        print(f"Tidak ada gambar di folder: {path}")
        continue

    for j in range(3):  # ambil 3 gambar
        img_path = os.path.join(path, random.choice(images))
        
        try:
            img = Image.open(img_path)

            plt.subplot(3, 3, i*3 + j + 1)
            plt.imshow(img)
            plt.axis('off')

            # Judul hanya di gambar pertama tiap baris
            if j == 0:
                plt.title(classes[i], fontsize=12, loc='left')

        except Exception as e:
            print(f"Error membuka gambar: {img_path}")
            print(e)

plt.tight_layout()

# === Simpan gambar (PENTING untuk skripsi) ===
plt.savefig('contoh_dataset.png', dpi=300)

plt.show()