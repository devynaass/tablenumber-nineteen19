# Laporan Hasil Persiapan Dataset & 30 Kartu Susunan Alternatif

## 1. Sumber Dataset
Dataset yang digunakan berasal dari kompetisi Kaggle: **Hardware Vision Challenge: Tabletop Object Counting**. Dataset diunduh dalam bentuk file ZIP dan diekstrak ke struktur direktori proyek di `data/raw/hardware-vision-challenge/`.

## 2. Karakteristik & Jumlah Data
* **Data Latih (Train Images)**: 1.600 citra meja (*tabletop*) berformat PNG (`1.png` - `1600.png`).
* **Masker Segmentasi (Train Masks)**: 1.600 citra masker anotasi piksel untuk data latih.
* **Data Uji (Test Images)**: 400 citra meja uji tanpa label (`1601.png` - `2000.png`).
* **Label Anotasi**: `train.csv` memuat kolom `id`, `books`, dan `glasses`.

## 3. Jumlah Kelas Objek
Dataset ini hanya memiliki **2 kelas objek**:
1. **Buku (`books`)**: Rentang nilai 0 sampai 5 objek per citra.
2. **Gelas / Kacamata (`glasses`)**: Rentang nilai 0 sampai 5 objek per citra.

## 4. Proses Pemilihan 30 Gambar Alternatif
Pemilihan 30 kartu dilakukan menggunakan skrip `src/select_arrangements.py`:
* Menyeleksi pasangan kombinasi unik antara jumlah buku dan gelas dari data latih ($6 \times 6 = 36$ kemungkinan pasangan).
* Memilih 30 kombinasi pertama yang merepresentasikan keberagaman konfigurasi objek (dari susunan minim objek hingga susunan padat 5 buku dan 5 gelas).
* Hasilnya dipetakan menjadi **Kartu 01** sampai **Kartu 30** sebagai kartu referensi susunan.

## 5. Keterbatasan Dataset
* **Cakupan Objek Terbatas**: Dataset Kaggle ini hanya memuat 2 jenis objek (`books` dan `glasses`), sehingga belum memenuhi skenario awal yang membutuhkan 14 kelas objek tabletop.
* **Sifat Alternatif**: Kartu susunan yang dihasilkan saat ini berfungsi sebagai **kartu referensi alternatif** untuk menguji alur kerja (*pipeline*) deteksi, penghitungan objek, dan tata letak spasial.

## 6. Kendala File Simulasi Resmi
* File simulasi resmi (yang memuat 14 objek lengkap dan lingkungan simulasi fisik 3D) saat ini **belum tersedia / belum dirilis secara resmi**.
* Oleh karena itu, 30 kartu susunan alternatif berbasis dataset Kaggle ini digunakan sebagai basis awal gladi rencana dan pengujian protokol model agar alur kerja tim tetap dapat berjalan.