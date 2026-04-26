import pickle
import matplotlib.pyplot as plt
import os

# path ke file history
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(BASE_DIR, "model", "riwayat.pkl")

# load history
with open(HISTORY_PATH, "rb") as f:
    history_data = pickle.load(f)

# buat range epoch
epochs_range = range(1, len(history_data["accuracy"]) + 1)

# plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs_range, history_data["accuracy"])
plt.plot(epochs_range, history_data["val_accuracy"])
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs_range) 
plt.ylim(0.75, 1.00) 
plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])                               
plt.legend(["Training", "Validation"])

plt.subplot(1,2,2)
plt.plot(epochs_range, history_data["loss"])
plt.plot(epochs_range, history_data["val_loss"])
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs_range)         
plt.legend(["Training", "Validation"])

plt.tight_layout()
plt.show()