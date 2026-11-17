# Catatan Sesi Tim 19: Deteksi dan Hitung Objek di Meja

## 1. Sesi yang Dipakai

| Butir | Isi |
|---|---|
| Tanggal slot | Senin, 16 November 2026, 16.00–17.00 |
| Folder sesi | `data/tim19/sesi-20261116-165743/` |
| Rencana | `rencana/t19-objek-meja.yaml` (seed 1019) |
| SHA-256 `label.csv` | lihat `data/MANIFEST.json` |

## 2. Pemeriksaan Mutu

- [ ] Jumlah kondisi di `label.csv` = 120 (30 susunan × 2 taraf lampu × 2 ulangan)
- [ ] Citra awal, tengah, dan akhir sesi tidak gelap, tidak buram, bingkai tidak bergeser, dan tidak ada tangan atau orang di dalam bingkai
- [ ] Isi citra cocok dengan kartu susunan (sampel acak). Temuan: S27 tidak sesuai, lihat bagian 3
- [ ] Lux BH1750 pada lampu 255 lebih tinggi daripada lampu 0
- [ ] Gangguan yang tercatat di lembar sesi sudah dicocokkan dengan `label.csv`

## 3. Kejanggalan: Susunan S27 Tidak Sesuai Rencana

| Butir | Rencana | Yang ditata saat sesi |
|---|---|---|
| Daftar objek | 2 ponsel, buku | 1 ponsel, 1 buku |
| Vektor label | `[1, 0, 2, 0, 0, 0, 0, 0]` | `[1, 0, 1, 0, 0, 0, 0, 0]` |

Urutan kelas pada vektor: buku, botol, ponsel, gunting, spidol, gelas, kotak pensil, kunci.

- **Kondisi terdampak:** keempat kondisi pada blok S27 (2 taraf lampu × 2 ulangan), yaitu semua baris `label.csv` dengan susunan S27.
- **Selisih:** kurang 1 ponsel. Kelas lain sesuai rencana.
- **Ditemukan:** 17 November 2026, setelah sesi.
- **Bukti:** citra kamera pada keempat kondisi S27. Foto cadangan ponsel: `[isi path]`.
- **Dampak bila dibiarkan:** label dari desain (2 ponsel) tidak sesuai dengan isi citra (1 ponsel), sehingga kelas ponsel pada S27 dilatih atau diuji dengan label yang salah.

## 4. Penanganan

**Usulan:** label S27 dikoreksi sesuai isi yang benar-benar ditata (1 buku, 1 ponsel). Data tidak dibuang karena jumlah susunan hanya 30.

- `label.csv` asli tidak diubah, sehingga sidik SHA-256 di `data/MANIFEST.json` tetap berlaku.
- Koreksi diterapkan di pemotong jendela melalui berkas `data/koreksi_label.csv`:

  ```csv
  susunan,kelas,jumlah_rencana,jumlah_sebenarnya
  S27,ponsel,2,1
  ```

- Pembagian lipatan tidak berubah; S27 tetap menjadi satu grup.
- Deviasi ini disebutkan dalam laporan.

**Alternatif** bila pembimbing tidak menyetujui koreksi: keempat kondisi S27 dikeluarkan, sehingga tersisa 29 susunan (satu lipatan berisi 9 susunan).

**Status:** menunggu persetujuan pembimbing (tanggal: `[isi]`).

## 5. Kondisi yang Dikeluarkan

| Susunan / nomor kondisi | Alasan |
|---|---|
| Belum ada | |

Gangguan lain yang tercatat di lembar sesi: tidak ada
