# Protokol Evaluasi Tim — Deteksi dan Penghitungan Objek pada Meja

## 1. Identitas Kasus

| Butir | Isi |
|---|---|
| Kasus | Deteksi dan penghitungan objek pada meja menggunakan citra kamera |
| Sumber dataset | Data primer hasil pengambilan citra kamera pada area meja A3 |
| Jenis tugas (`TUGAS`) | Mengikuti kontrak starter yang telah disepakati tim |
| Target / label | Jumlah objek dari 8 kelas |
| Jumlah kelas | 8 kelas objek |
| Jumlah susunan | 30 susunan |
| Kondisi pencahayaan | 2 taraf |
| Pengulangan | 2 kali |
| Total kondisi | 120 kondisi |

## 2. Kelas Objek

Objek yang dihitung terdiri atas:

1. Buku
2. Botol
3. Ponsel
4. Gunting
5. Spidol
6. Gelas
7. Kotak pensil
8. Kunci

Target model berupa vektor jumlah objek untuk masing-masing kelas.

## 3. Karakteristik Data

| Komponen | Ketentuan |
|---|---|
| Sumber citra | Kamera |
| Resolusi awal | 1280 × 720 piksel |
| Interval pengambilan | 0,5 detik |
| Area pemrosesan | Area meja A3 |
| Ukuran citra masukan | 224 × 224 piksel |
| Periode awal yang dibuang | 1 detik pertama setiap kondisi |
| Format label | Mengikuti `label.csv` |

## 4. Prapemrosesan Data

1. Mengambil citra dari folder kamera.
2. Memotong citra berdasarkan area meja A3.
3. Mengubah ukuran citra menjadi 224 × 224 piksel.
4. Membuang frame pada detik 0 dan 0,5.
5. Melakukan normalisasi berdasarkan statistik data latih saja.
6. Menerapkan prapemrosesan yang sama pada data latih, validasi, dan uji.
7. Tidak menggunakan augmentasi flip yang mengubah susunan objek.

## 5. Pembagian Data

Pembagian data dilakukan dengan memperhatikan identitas susunan objek untuk mencegah kebocoran data.

- Data dari satu susunan tidak boleh tersebar pada bagian data latih dan uji.
- Pembagian train dan validation mengikuti `src.common.split` dengan seed 2026, setelah unit pembagian ditetapkan.
- Pembagian data uji mengikuti ketentuan 3 fold, masing-masing terdiri atas 10 susunan.
- Detail penetapan data validasi dan mekanisme fold dikonfirmasi bersama tim dan pembimbing.

## 6. Metrik Evaluasi

Metrik utama untuk membandingkan model:

1. MAE untuk masing-masing dari 8 kelas objek.
2. Ketepatan penghitungan susunan.

Metrik tambahan:

- Ukuran model ONNX.
- Latensi inferensi pada laptop.

Hasil komparasi dilaporkan menggunakan rata-rata dan simpangan baku dari 3 seed.

## 7. Pengelolaan Data

Data primer yang diterima meliputi:

- `label.csv`
- CSV sensor
- `manifest.json`
- Citra kamera pada penyimpanan tim.

Penjaga Data & Evaluasi bertanggung jawab untuk:

1. Memeriksa kelengkapan data.
2. Memastikan terdapat 120 kondisi atau mendokumentasikan ketidaksesuaian.
3. Memeriksa kesesuaian citra dengan kartu susunan.
4. Mengisi `data/MANIFEST.json`.
5. Memastikan `data_hash` konsisten di seluruh anggota tim.
6. Mencatat sesi yang digunakan dan kondisi yang dikecualikan dalam `data/CATATAN-SESI.md`.

## 8. Evaluasi Data Uji

Data uji hanya dievaluasi satu kali oleh Penjaga Data pada sprint terakhir menggunakan opsi `--final`.

Prediksi dan hasil evaluasi disimpan dalam direktori `results/`.

## 9. Komputasi dan Reproduksibilitas

- Seed pelatihan mengikuti seed yang ditentukan pada setiap eksperimen.
- Seed pembagian data mengikuti seed 2026.
- Perangkat komputasi dicatat dalam hasil evaluasi.
- Durasi pelatihan dicatat dalam `metrics.json`.
- Hasil evaluasi menggunakan format keluaran standar Workbench.

## 10. Persetujuan dan Pembekuan Protokol

Disetujui pembimbing pada: [diisi setelah persetujuan]

Tautan Pull Request: [diisi setelah PR dibuat]

`data_hash` saat dibekukan: [diisi setelah baseline tersedia]