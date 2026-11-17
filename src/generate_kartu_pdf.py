import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PIL import Image

# Path berkas
TRAIN_CSV = "data/raw/hardware-vision-challenge/train.csv"
IMG_DIR = "data/raw/hardware-vision-challenge/train_data/train_images"
OUTPUT_PDF = "docs/kartu/30_kartu_susunan.pdf"

os.makedirs("docs/kartu", exist_ok=True)

# 1. Ambil 30 kombinasi unik
df = pd.read_csv(TRAIN_CSV)
selected_30 = (
    df.drop_duplicates(subset=["books", "glasses"])
    .head(30)
    .reset_index(drop=True)
)

print("Sedang menyusun 30_kartu_susunan.pdf...")

# 2. Buat dokumen PDF multi-halaman
with PdfPages(OUTPUT_PDF) as pdf:
    for idx, row in selected_30.iterrows():
        img_id = int(row["id"])
        books = int(row["books"])
        glasses = int(row["glasses"])
        card_no = f"{idx+1:02d}"

        img_path = os.path.join(IMG_DIR, f"{img_id}.png")

        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # Ukuran kertas A4 Portrait

        # Judul & Informasi Kartu
        plt.suptitle(
            f"KARTU SUSUNAN REFERENSI #{card_no}",
            fontsize=18,
            fontweight="bold",
            y=0.94,
        )

        info_text = (
            f"ID Gambar: {img_id}   |   Jumlah Buku: {books}   |   Jumlah Gelas: {glasses}\n"
            f"Keterangan: Kartu Alternatif Gladi Rencana (Hardware Vision Challenge)"
        )
        ax.set_title(info_text, fontsize=11, pad=15)

        # Muat dan tampilkan gambar referensi
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis("on")
            ax.set_xlabel("Koordinat Piksel (X)", fontsize=9)
            ax.set_ylabel("Koordinat Piksel (Y)", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                f"Gambar {img_id}.png tidak ditemukan",
                ha="center",
                va="center",
            )

        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])
        pdf.savefig(fig)
        plt.close(fig)

print(f"Berhasil! File PDF tersimpan di: {OUTPUT_PDF}")