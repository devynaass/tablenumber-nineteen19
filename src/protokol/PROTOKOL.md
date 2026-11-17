# Protokol Evaluasi Tim — Deteksi dan Penghitungan Objek pada Meja

## 1. Identitas Kasus

| Butir | Isi |
|---|---|
| Kasus | Deteksi dan penghitungan objek pada meja menggunakan citra kamera |
| Sumber dataset | Data primer sesi `sesi-20261116-165743` pada area meja A3 |
| Jenis tugas (`TUGAS`) | `"regresi"` |
| Target / label | Vektor jumlah objek dari 8 kelas (`float32 [N, 8]`) |
| Jumlah kelas | 8 kelas objek |
| Jumlah susunan | 30 susunan (`S01`–`S30`) |
| Kondisi pencahayaan | 2 taraf (lampu `0` dan `255`) |
| Pengulangan | 2 kali |
| Total kondisi | 120 kondisi (dengan koreksi label `S27`: `ponsel = 1`, sesuai `data/CATATAN-SESI.md`) |

## 2. Kelas Objek

Target model berupa vektor 8 dimensi (`float32 [N, 8]`) dengan urutan indeks baku yang identik dengan `KELAS_OBJEK` pada `src/protokol/data.py`:

1. `botol` (indeks 0)
2. `gelas` (indeks 1)
3. `kotak pensil` (indeks 2)
4. `kunci` (indeks 3)
5. `spidol` (indeks 4)
6. `gunting` (indeks 5)
7. `ponsel` (indeks 6)
8. `buku` (indeks 7)

## 3. Karakteristik Data

| Komponen | Ketentuan |
|---|---|
| Sumber citra | Kamera (`data/raw/kamera/`) |
| Resolusi awal | 1280 × 720 piksel |
| Interval pengambilan | 0,5 detik |
| Area pemrosesan (`CROP_A3`) | Area meja A3 pada koordinat piksel `(kiri, atas, kanan, bawah) = [ISI_4_ANGKA_CROP_A3]` |
| Ukuran citra masukan | 224 × 224 piksel (`[3, 224, 224]`, `float32` skala `0..1`) |
| Periode awal yang dibuang | 1 detik pertama setiap kondisi ($t\_mulai\_host + 1.0 \le t\_host \le t\_selesai\_host$) |
| Format label | Diekstrak dari kolom `susunan` pada `label.csv` |

## 4. Prapemrosesan Data

1. Mengambil citra dari `data/raw/kamera/` berdasarkan rentang waktu tiap kondisi pada `label.csv` setelah membuang 1 detik pertama (frame pada detik 0 dan 0,5).
2. Memotong citra (*crop*) pada area meja A3 menggunakan koordinat baku `CROP_A3 = [173, 116, 611, 355]`.
3. Mengubah ukuran (*resize*) citra menjadi 224 × 224 piksel RGB (`BILINEAR`) dengan skala piksel `0..1` (`[C, H, W]`).
4. Melakukan normalisasi statistik (mean/std) di dalam model menggunakan statistik data latih saja.
5. Menerapkan prapemrosesan yang sama pada data latih, validasi, dan uji.
6. Tidak menggunakan augmentasi *flip* yang mengubah posisi spasial susunan objek.

## 5. Pembagian Data

Pembagian data dilakukan per satuan **ID susunan** (`S01`–`S30`) untuk mencegah kebocoran data (*data leakage*):

- Data dari satu susunan tidak pernah tersebar pada bagian data latih, validasi, dan uji secara bersamaan.
- 30 susunan diacak menggunakan `seed=2026` lalu dibagi menjadi **3 lipatan (*fold*) × 10 susunan** (`LIPATAN=0, 1, 2` sebagai himpunan uji).
- Dari 20 susunan sisa (non-uji), dibagi melalui `src.common.split` (`porsi_val=0.2`, `seed=2026`) menjadi **16 susunan latih** dan **4 susunan validasi** (menghasilkan 384 sampel latih pada `LIPATAN=0`).

## 6. Metrik Evaluasi

Metrik utama untuk membandingkan model:

1. **MAE per kelas** (*Mean Absolute Error* untuk masing-masing dari 8 kelas objek).
2. **Ketepatan penghitungan susunan** (*exact match accuracy* setelah pembulatan prediksi ke bilangan bulat terdekat untuk seluruh 8 kelas sekaligus).

Metrik tambahan:

- Ukuran model ONNX.
- Latensi inferensi pada laptop.

Hasil komparasi dilaporkan menggunakan rata-rata dan simpangan baku dari 3 seed.

## 7. Pengelolaan Data

Data primer yang diterima meliputi `label.csv`, CSV sensor, `manifest.json`, dan citra kamera pada penyimpanan tim.

Penjaga Data & Evaluasi bertanggung jawab untuk:

1. Memeriksa kelengkapan data dan memastikan 120 kondisi sesuai rencana.
2. Memeriksa kesesuaian citra dengan kartu susunan pada beberapa sampel acak.
3. Mengisi `data/MANIFEST.json` dan memastikan `data_hash` konsisten di seluruh anggota tim.
4. Mencatat sesi yang digunakan serta koreksi kondisi/label dalam `data/CATATAN-SESI.md`.

## 8. Evaluasi Data Uji

Data uji hanya dievaluasi satu kali oleh Penjaga Data pada sprint terakhir menggunakan opsi `--final`. Prediksi dan hasil evaluasi disimpan dalam direktori `results/`.

## 9. Komputasi dan Reproduksibilitas

- Seed pelatihan mengikuti seed yang ditentukan pada setiap eksperimen; seed pembagian data dikunci pada `2026`.
- Perangkat komputasi dicatat dalam hasil evaluasi dan durasi pelatihan dicatat dalam `metrics.json` menggunakan format keluaran standar Workbench.

## 10. Persetujuan dan Pembekuan Protokol

Disetujui pembimbing pada: [diisi setelah persetujuan]

Tautan Pull Request: [diisi setelah PR dibuat]

`data_hash` saat dibekukan: [diisi dari data/MANIFEST.json]