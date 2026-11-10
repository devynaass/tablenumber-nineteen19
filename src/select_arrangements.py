import os
import pandas as pd

# 1. Tentukan path dataset
DATA_PATH = "data/raw/hardware-vision-challenge/train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Baca file train.csv
df = pd.read_csv(DATA_PATH)

# 3. Ambil 1 sampel gambar untuk setiap pasangan kombinasi unik (books, glasses)
# Total kombinasi unik adalah 36 (dari 0-5 buku x 0-5 gelas)
unique_combinations = df.drop_duplicates(
    subset=["books", "glasses"]
).reset_index(drop=True)

# 4. Pilih 30 kombinasi pertama yang paling bervariasi
selected_30 = unique_combinations.head(30).copy()

# 5. Format kolom menjadi kartu susunan (Kartu 01 s.d. 30)
selected_30["Kartu"] = [f"{i+1:02d}" for i in range(len(selected_30))]
selected_30 = selected_30.rename(
    columns={"id": "ID gambar", "books": "Buku", "glasses": "Gelas"}
)

# Susun urutan kolom yang rapi
hasil_kartu = selected_30[["Kartu", "ID gambar", "Buku", "Gelas"]]

# 6. Simpan hasil ke file CSV untuk dokumentasi
output_file = os.path.join(OUTPUT_DIR, "30_kartu_susunan.csv")
hasil_kartu.to_csv(output_file, index=False)

# 7. Tampilkan tabel 30 susunan di terminal
print("=" * 45)
print("     30 PILIHAN KARTU SUSUNAN ALTERNATIF    ")
print("=" * 45)
print(hasil_kartu.to_string(index=False))
print("=" * 45)
print(f"Hasil tersimpan di: {output_file}")
print("\nCatatan Verifikasi Visual:")
print("Cek gambar di: data/raw/hardware-vision-challenge/train_data/train_images/")
print(
    "Pastikan objek tidak bertumpuk dan berada dalam batas area meja/A3 untuk ID di atas."
)