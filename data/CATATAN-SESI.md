# Catatan Sesi Tim 19: Deteksi dan Hitung Objek di Meja

## 1. Sesi yang Dipakai

| Butir | Isi |
|---|---|
| Tanggal slot | Senin, 16 November 2026, 16.00–17.00 |
| Folder sesi | `data/raw/sesi-20261116-165743/` |
| Rencana | `rencana/t19-objek-meja.yaml` (seed 1019) |
| SHA-256 `label.csv` / `data_hash` | `366c08fe1dd6031eb728073288512de8` (lihat `data/MANIFEST.json`) |

## 2. Pemeriksaan Mutu

- [x] Jumlah kondisi di `label.csv` = 120 (30 susunan × 2 taraf lampu × 2 ulangan)
- [x] Citra awal, tengah, dan akhir sesi tidak gelap, tidak buram, bingkai tidak bergeser, dan tidak ada tangan atau orang di dalam bingkai
- [x] Isi citra cocok dengan kartu susunan (sampel acak). Temuan: S27 tidak sesuai, lihat bagian 3 & 6
- [x] Lux BH1750 pada lampu 255 dan 0 tercatat konsisten pada setiap kondisi
- [x] Gangguan yang tercatat di lembar sesi sudah dicocokkan dengan `label.csv`

## 3. Kejanggalan: Susunan S27 Tidak Sesuai Rencana

| Butir | Rencana | Yang ditata saat sesi |
|---|---|---|
| Daftar objek | 2 ponsel, buku | 1 ponsel, 1 buku |
| Vektor label | `[0, 0, 0, 0, 0, 0, 2, 1]` | `[0, 0, 0, 0, 0, 0, 1, 1]` |

Urutan kelas pada vektor mengikuti `KELAS_OBJEK` di `src/protokol/data.py`: botol, gelas, kotak pensil, kunci, spidol, gunting, ponsel, buku.

- **Kondisi terdampak:** keempat kondisi pada blok S27 (2 taraf lampu × 2 ulangan), yaitu semua baris `label.csv` dengan susunan S27.
- **Selisih:** kurang 1 ponsel. Kelas lain sesuai rencana.
- **Ditemukan:** 17 November 2026, setelah sesi.
- **Bukti:** citra kamera pada keempat kondisi S27 (`data/raw/kamera/`).
- **Dampak bila dibiarkan:** label dari desain (2 ponsel) tidak sesuai dengan isi citra (1 ponsel), sehingga kelas ponsel pada S27 dilatih atau diuji dengan label yang salah.

## 4. Penanganan

**Usulan:** label S27 dikoreksi sesuai isi yang benar-benar ditata (1 buku, 1 ponsel). Data tidak dibuang karena jumlah susunan hanya 30.

- `label.csv` asli tidak diubah, sehingga sidik SHA-256 di `data/MANIFEST.json` tetap berlaku.
- Koreksi diterapkan di `src/protokol/data.py` melalui `KOREKSI_LABEL = {"S27": {"ponsel": 1.0}}`. Koreksi ditaruh di kode (bukan di `data/`) karena folder `data/` tidak di-commit, sehingga koreksi ini sama di laptop semua anggota.
- Pembagian lipatan tidak berubah; S27 tetap menjadi satu grup.
- Deviasi ini disebutkan dalam laporan.

**Alternatif** bila pembimbing tidak menyetujui koreksi: keempat kondisi S27 dikeluarkan dengan `SUSUNAN_DIBUANG = {"S27"}` dan entri S27 dihapus dari `KOREKSI_LABEL`, sehingga tersisa 29 susunan (satu lipatan berisi 9 susunan).

**Status:** diterapkan dalam kode protokol dan diajukan untuk persetujuan pembimbing.

## 5. Kondisi yang Dikeluarkan

| Susunan / nomor kondisi | Alasan |
|---|---|
| Tidak ada (0 kondisi dibuang) | Seluruh 120 kondisi layak pakai setelah koreksi label S27; 1 detik awal tiap kondisi (`t_mulai_host + 1.0`) dibuang otomatis saat pemuatan data |

Gangguan lain yang tercatat di lembar sesi: tidak ada.

## 6. Bukti Pemeriksaan Sampel Citra & Keseragaman `data_hash`

### A. Hasil Pemeriksaan Silang Citra Kamera terhadap Kartu Susunan
Pemeriksaan visual dilakukan terhadap sampel acak kondisi dari `label.csv` dengan mencocokkan berkas citra pada rentang `[t_mulai_host + 1.0, t_selesai_host]` di `data/raw/kamera/`:

| No. Kondisi | Kartu Susunan (`label.csv`) | Taraf Lampu / Lux | Berkas Citra (`kamera/`) | Hasil Inspeksi Visual |
| :---: | :--- | :---: | :--- | :--- |
| `1` | `S16: botol, gelas, 2 kotak pensil, kunci, spidol` | `0` (`221.7 lux`) | `f000076.jpg` (`f000075`–`f000081`) | Cocok (6 objek sesuai kartu pada area A3) |
| `5` | `S22: botol, gunting` | `255` (`54.2 lux`) | `f000230.jpg` (`f000229`–`f000235`) | Cocok (1 botol & 1 gunting sesuai kartu) |
| `9` | `S14: kunci` | `0` (`220.0 lux`) | `f000324.jpg` (`f000323`–`f000329`) | Cocok (1 kunci sesuai kartu) |
| `S27` | `S27` (rencana `2 ponsel` -> aktual `1 ponsel`) | `0` & `255` | Seluruh frame blok `S27` | Terverifikasi `1 ponsel` & `1 buku` pada citra; koreksi diterapkan |

### B. Ringkasan Kecocokan `data_hash` Seluruh Anggota Tim
- **Nilai `data_hash` Acuan (`data/MANIFEST.json`):** `366c08fe1dd6031eb728073288512de8`

| NIM | Nama Anggota | `data_hash` Terpakai | Status |
| :---: | :--- | :---: | :---: |
| `123450007` | Juesia Pridelia Saragih | `366c08fe1dd6031eb728073288512de8` | Cocok |
| `123450064` | Muhammad Hanif Dzaky | `366c08fe1dd6031eb728073288512de8` | Cocok |
| `123450075` | Aliyammara | `366c08fe1dd6031eb728073288512de8` | Cocok |
| `123450085` | Devyna Sonya Palupi Sanjaya | `366c08fe1dd6031eb728073288512de8` | Cocok |
| `123450090` | Malika Azzahra Salsabila | `366c08fe1dd6031eb728073288512de8` | Cocok |