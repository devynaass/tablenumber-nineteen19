# Catatan Jalur CNN (NIM 123450007)

Semua angka diambil dari `riwayat_latih.csv` dan log pelatihan. Data: `LIPATAN=0` (16 susunan latih / 384 citra, 4 susunan validasi / 96 citra). Data uji belum disentuh.

## 1. Arsitektur

CNN kecil dari nol, masukan 3 × 224 × 224, keluaran 8 angka jumlah objek (Softplus, tidak negatif).

| Tahap | Lapisan | Ukuran keluaran |
|---|---|---|
| Stem | Conv 3×3 stride 2, BN, ReLU (32 filter) | 112 × 112 |
| Blok 1 | Conv, BN, ReLU, MaxPool (32) | 56 × 56 |
| Blok 2 | 2 × (Conv, BN, ReLU), MaxPool (64) | 28 × 28 |
| Blok 3 | 2 × (Conv, BN, ReLU), MaxPool (128) | 14 × 14 |
| Kepala | Conv, BN, ReLU, Global Average Pooling, Dropout, Linear 64, ReLU, Linear 8, Softplus | 8 |

Jumlah parameter: 443.944 (`--lebar 32`); 113.976 untuk `--lebar 16`.

## 2. Optimizer dan alasannya

AdamW, lr 1e-3, weight decay 1e-4, jadwal cosine selama 40 epoch, batch 32.

Alasan: laju belajar adaptif per parameter membuat AdamW cepat konvergen tanpa penyetelan lr yang panjang, penting karena anggaran ≤ 20 menit per run di CPU. Weight decay pada AdamW dipisahkan dari gradien sehingga berfungsi sebagai regularisasi yang sebenarnya (berbeda dengan L2 pada Adam biasa). Jadwal cosine menurunkan lr menjelang akhir agar bobot lebih stabil.

## 3. Regularisasi

- Weight decay (AdamW)
- Dropout 0,3 sebelum lapisan linear
- Augmentasi data latih: kontras 0,7–1,3 dan kecerahan ±0,1 (meniru lampu 0 dan 255), geser acak ±8 piksel. Tanpa flip sesuai protokol.
- Berhenti dini: 15 epoch tanpa perbaikan MAE validasi; bobot terbaik dipakai

## 4. Patokan pembanding (validasi `LIPATAN=0`)

| Tebakan | val_mae |
|---|---|
| Selalu nol untuk semua kelas | 0,219 |
| Selalu rata-rata data latih | 0,469 |

Validasi hanya berisi 4 susunan dengan 1 sampai 3 objek (S01, S12, S23, S29) dan tidak memuat botol maupun gelas, sehingga tebakan nol sudah menghasilkan MAE rendah. Karena itu MAE harus dibaca bersama ketepatan susunan dan MAE per kelas.

## 5. Pencarian hyperparameter

| Run | Konfigurasi | val_mae terbaik | Epoch terbaik | Ketepatan susunan | Waktu latih (s) |
|---|---|---|---|---|---|
| 1 | lr 1e-3, lebar 32, dropout 0,3, wd 1e-4, MSE | 0,2772 | 10 | 0 | 764 |
| 2 | lebar 16, dropout 0,5, wd 1e-3, MSE | 0,2848 | 17 | 0 | 351 |
| 3 | lr 3e-4, MSE | 0,3563* | 12 | 0 | 1115 |
| 4 | lr 1e-3, lebar 32, dropout 0,3, wd 1e-4, **L1** | **0,2278** | 21 | 0 | 1087 |

\* Run 3 tidak sah: satu epoch tertahan sekitar 13 menit (laptop tidur atau CPU diperlambat), anggaran 18 menit habis saat MAE validasi masih turun.

Run 1 diulang dengan seed yang sama dan menghasilkan angka identik (0,27716 pada epoch 10), sehingga pelatihan dapat direproduksi.

Model lebih kecil dengan regularisasi lebih kuat (run 2) mengurangi overfitting pada data latih (loss latih akhir 0,08 dibanding 0,03) tetapi tidak memperbaiki MAE validasi.

Konfigurasi terpilih: run 4 (loss L1), karena MAE validasinya terendah dan kurvanya konsisten lebih rendah sepanjang pelatihan (sekitar 0,23–0,28, dibanding 0,28–0,35 pada MSE), bukan hanya pada satu epoch. L1 juga mengoptimalkan langsung MAE, metrik utama kasus.

## 6. Ablation

| Varian | Perintah | val_mae | Ketepatan susunan |
|---|---|---|---|
| Konfigurasi terpilih (L1, augmentasi) | `--seed 1` | 0,2278 | 0 |
| Loss MSE (bukan L1) | `--seed 1 --loss mse` | 0,2772 | 0 |
| Tanpa augmentasi (MSE) | `--seed 1 --loss mse --tanpa-aug` | 0,2796 | 0 |

MAE validasi per kelas:

| Kelas | L1 (terpilih) | MSE | MSE tanpa augmentasi | Tebak nol |
|---|---|---|---|---|
| Botol | 0,101 | 0,307 | 0,159 | 0,000 |
| Gelas | 0,007 | 0,035 | 0,051 | 0,000 |
| Kotak pensil | 0,254 | 0,278 | 0,278 | 0,250 |
| Kunci | 0,549 | 0,596 | 0,553 | 0,500 |
| Spidol | 0,318 | 0,233 | 0,321 | 0,250 |
| Gunting | 0,293 | 0,315 | 0,406 | 0,250 |
| Ponsel | 0,073 | 0,171 | 0,186 | 0,250 |
| Buku | 0,228 | 0,283 | 0,285 | 0,250 |

Kesimpulan:

- Augmentasi tidak memberi perbedaan terukur pada validasi (0,2772 dibanding 0,2796); selisih ini masih dalam rentang naik-turun validasi 4 susunan. Tanpa augmentasi, loss latih turun lebih rendah (0,017), tanda model lebih menghafal.
- Loss L1 menurunkan MAE validasi paling besar (0,2772 menjadi 0,2278). Pada ponsel dan buku model L1 lebih baik dari tebakan nol, tanda ada pola yang dipelajari.
- Tidak ada konfigurasi yang mencapai ketepatan susunan di atas 0 atau mengalahkan tebakan nol secara keseluruhan (0,219). Kunci adalah kelas tersulit di semua konfigurasi; dugaan awal karena ukurannya paling kecil pada citra 224 × 224 (perlu dicek pada `alat/contoh_crop.png`).
- Dengan hanya 16 susunan latih, CNN dari nol belum cukup untuk belajar menghitung secara andal. Hasil ini menjadi patokan bagi jalur transfer learning.

## 7. Hasil 3 seed (konfigurasi akhir)

Dijalankan pada Sprint 2:

```
python models/123450007/run.py --seed 1
python models/123450007/run.py --seed 2
python models/123450007/run.py --seed 3
```

| Seed | val_mae | Ketepatan susunan | Durasi (s) |
|---|---|---|---|
| 1 | 0,2278 | 0 | |
| 2 | | | |
| 3 | | | |
| Rata-rata ± simpangan | | | |

Ukuran ONNX (KB) dan latensi (ms) di laptop meja: [isi pada Sprint 2]