import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Path
CSV_PATH = "data/raw/hardware-vision-challenge/train.csv"
IMG_DIR = "data/raw/hardware-vision-challenge/train_data/train_images"
OUTPUT_DIR = "outputs/analisis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Ambil 6 sampel pertama
samples = df.head(6)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (_, row) in enumerate(samples.iterrows()):
    img_id = int(row["id"])
    books = int(row["books"])
    glasses = int(row["glasses"])

    img_path = os.path.join(IMG_DIR, f"{img_id}.png")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"ID: {img_id}\nBuku: {books} | Gelas: {glasses}", fontsize=10
        )
    axes[idx].axis("off")

plt.suptitle("Sampel Citra Tabletop Object Counting", fontsize=14, y=0.98)
plt.tight_layout()
output_img = os.path.join(OUTPUT_DIR, "sample_images_grid.png")
plt.savefig(output_img, dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualisasi sampel berhasil disimpan di: {output_img}")