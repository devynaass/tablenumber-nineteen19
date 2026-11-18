"""src/simulasi_data.py

Mengambil sampel gambar asli Kaggle (Hardware Vision Challenge),
mengubah ukuran ke 224x224 (RGB), dan menyimpannya ke data/simulasi/
untuk pengujian model CNN-LSTM (Jalur RNN).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

# Path direktori
BASE_DIR = "data/raw/hardware-vision-challenge"
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
IMG_DIR = os.path.join(BASE_DIR, "train_data", "train_images")
OUT_DIR = "data/simulasi"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Baca data label CSV
df = pd.read_csv(CSV_PATH)

# Ambil 80 data latih dan 20 data validasi agar simulasi cepat di laptop
N_TRAIN = 80
N_VAL = 20
TOTAL = N_TRAIN + N_VAL
df_sample = df.head(TOTAL)

print(f"Mengolah {TOTAL} gambar asli Kaggle menjadi 224x224...")


def proses(sub_df):
    X, y = [], []
    for _, row in sub_df.iterrows():
        img_id = int(row["id"])
        label = int(row["books"])  # Target jumlah buku (0 s.d. 5)

        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img_resized = img.convert("RGB").resize(
                    (224, 224), Image.Resampling.BILINEAR
                )
                arr = (
                    np.array(img_resized, dtype=np.float32) / 255.0
                )  # Normalisasi 0-1
                arr = np.transpose(arr, (2, 0, 1))  # Ubah jadi (C, H, W)
                X.append(arr)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# 2. Proses data latih & validasi
X_train, y_train = proses(df_sample.iloc[:N_TRAIN])
X_val, y_val = proses(df_sample.iloc[N_TRAIN:])

# 3. Simpan ke data/simulasi/
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)

print("=" * 50)
print("DATA SIMULASI KAGGLE BERHASIL DISIAPKAN!")
print("=" * 50)
print(f"X_train : {X_train.shape} (80 citra 224x224 RGB)")
print(f"y_train : {y_train.shape} (label kelas 0-5)")
print(f"X_val   : {X_val.shape}   (20 citra 224x224 RGB)")
print(f"y_val   : {y_val.shape}")
print(f"Lokasi  : {OUT_DIR}/")