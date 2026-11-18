import os
import matplotlib.pyplot as plt
import pandas as pd

# 1. Gunakan path dari posisi file ini berada agar selalu tepat
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Folder capstone-deep-learning

BASE_DIR = os.path.join(
    PROJECT_ROOT, "data", "raw", "hardware-vision-challenge"
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "analisis")

# 2. Buat folder output otomatis jika belum ada
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. Baca data
train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

# Statistik dataset
summary = train[["books", "glasses"]].describe()
summary.to_csv(os.path.join(PROCESSED_DIR, "dataset_summary.csv"))

print("Jumlah gambar:", len(train))
print("Total buku:", train["books"].sum())
print("Total gelas:", train["glasses"].sum())

# Distribusi jumlah buku
train["books"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribusi Jumlah Buku")
plt.xlabel("Jumlah Buku")
plt.ylabel("Jumlah Gambar")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribution_books.png"))
plt.close()

# Distribusi jumlah gelas
train["glasses"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribusi Jumlah Gelas")
plt.xlabel("Jumlah Gelas")
plt.ylabel("Jumlah Gambar")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distribution_glasses.png"))
plt.close()

print("Analisis selesai.")