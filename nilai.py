from PIL import Image
import numpy as np

img = Image.open("gambar.jpg")
img = img.resize((5,5))

array = np.array(img)

# Normalisasi (0 - 255 → 0 - 1)
normalized = array / 255.0

print(normalized)