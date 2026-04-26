import os

base_dir = "dataset_split"
splits = ["train", "val", "test"]
image_ext = (".jpg", ".jpeg", ".png")

classes = sorted(os.listdir(os.path.join(base_dir, "train")))

print("📊 RINGKASAN PEMBAGIAN DATASET PER KELAS (70% : 15% : 15%)\n")

for kelas in classes:
    total = 0
    data = {}

    for split in splits:
        path = os.path.join(base_dir, split, kelas)
        jumlah = len([
            f for f in os.listdir(path)
            if f.lower().endswith(image_ext)
        ])
        data[split] = jumlah
        total += jumlah

    print(f"Kelas: {kelas}")
    print(f"  Train : {data['train']:4d} gambar ({data['train']/total*100:.2f}%)")
    print(f"  Val   : {data['val']:4d} gambar ({data['val']/total*100:.2f}%)")
    print(f"  Test  : {data['test']:4d} gambar ({data['test']/total*100:.2f}%)")
    print(f"  Total : {total} gambar\n")
