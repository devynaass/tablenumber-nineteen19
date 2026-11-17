# Catatan Jalur CNN (NIM 123450007)

Semua angka di bawah diambil dari `riwayat_latih.csv` (satu baris per run). Data: `LIPATAN=0`, hanya validasi.

## 1. Arsitektur

CNN kecil dari nol, masukan 3 × 224 × 224, keluaran 8 angka jumlah objek (Softplus, tidak negatif).

| Tahap | Lapisan | Ukuran keluaran |
|---|---|---|
| Stem | Conv 3×3 stride 2, BN, ReLU (`lebar` filter) | 112 × 112 |
| Blok 1 | Conv, BN, ReLU, MaxPool | 56 × 56 |
| Blok 2 | 2 × (Conv, BN, ReLU), MaxPool (2 × `lebar`) | 28 × 28 |
| Blok 3 | 2 × (Conv, BN, ReLU), MaxPool (4 × `lebar`) | 14 × 14 |
| Kepala | Conv, BN, ReLU, Global Average Pooling, Dropout, Linear 64, ReLU, Linear 8, Softplus | 8 |

Jumlah parameter: [isi, `sum(p.numel() for p in model.parameters())`]

## 2. Optimizer dan alasannya

AdamW, lr [isi], weight decay [isi], jadwal cosine. Alasan: [isi]

## 3. Regularisasi

- Weight decay (AdamW)
- Dropout sebelum lapisan linear
- Augmentasi data latih: kontras 0,7–1,3 dan kecerahan ±0,1 (meniru lampu 0 dan 255), geser acak ±8 piksel. Tanpa flip sesuai protokol.
- Berhenti dini berdasarkan MAE validasi

## 4. Pencarian hyperparameter

| Run | lr | lebar | dropout | loss | val_mae | ketepatan susunan | epoch terbaik |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |

Contoh perintah: `--lr 3e-4`, `--lr 3e-3`, `--lebar 16`, `--lebar 48`, `--dropout 0.5`.

Konfigurasi terpilih: [isi]

## 5. Ablation (minimal 2)

| Varian | Perintah | val_mae | ketepatan susunan |
|---|---|---|---|
| Konfigurasi terpilih | `--seed 1` | | |
| Tanpa augmentasi | `--seed 1 --tanpa-aug` | | |
| Loss L1 (bukan MSE) | `--seed 1 --loss l1` | | |
| Tanpa dropout | `--seed 1 --dropout 0` | | |

Kesimpulan: [isi]

## 6. Hasil 3 seed (konfigurasi akhir)

| Seed | val_mae | ketepatan susunan | durasi (s) |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| Rata-rata ± simpangan | | | |

MAE per kelas (rata-rata 3 seed): [isi dari riwayat_latih.csv]