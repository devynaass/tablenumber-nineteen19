"""Pemuat data kasus tim 19 (hitung objek di meja). MILIK TIM, dibekukan setelah disetujui pembimbing.

Kontrak starter: TUGAS dan muat() -> (X_train, y_train, X_val, y_val, X_test, y_test).

X : float32 [N, 3, 224, 224], skala 0..1 (normalisasi dilakukan di model, dari data latih saja)
y : float32 [N, 8], jumlah objek per kelas dengan urutan KELAS_OBJEK

Pembagian per susunan (satu susunan tidak pernah terbagi):
    30 susunan diacak (seed 2026) lalu dibagi 3 lipatan x 10 susunan
    uji      = lipatan ke-LIPATAN (variabel lingkungan LIPATAN, bawaan 0)
    validasi = PORSI_VAL dari 20 susunan sisanya, lewat src.common.split
    latih    = sisanya
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

from src.common import manifest, split

TUGAS = "regresi"

# Urutan baku 8 kelas objek
KELAS_OBJEK = [
    "botol",
    "gelas",
    "kotak pensil",
    "kunci",
    "spidol",
    "gunting",
    "ponsel",
    "buku",
]

SESI = "sesi-20261116-165743"
JUMLAH_LIPATAN = 3
PORSI_VAL = 0.2  # 20 susunan non-uji -> 4 susunan validasi, 16 susunan latih

# Koordinat crop area A3 pada citra 1280x720: (kiri, atas, kanan, bawah) dalam piksel.
# Ambil dari satu citra asli: buka di Paint, arahkan kursor ke sudut-sudut kertas A3,
# baca koordinat piksel di pojok kiri bawah jendela Paint.
CROP_A3: tuple[int, int, int, int] | None = None

# Susunan yang ditata berbeda dari rencana (lihat data/CATATAN-SESI.md).
# Nilai di sini menimpa jumlah dari rencana.
KOREKSI_LABEL: dict[str, dict[str, float]] = {
    "S27": {"ponsel": 1.0},  # rencana 2 ponsel, yang ditata saat sesi 1 ponsel
}

# Diisi dari lembar sesi bila ada gangguan.
KONDISI_DIBUANG: set[int] = set()   # nomor kondisi = urutan baris label.csv, mulai dari 1
SUSUNAN_DIBUANG: set[str] = set()   # mis. {"S27"} bila pembimbing memilih membuang S27


def _parse_susunan(teks_susunan: str) -> tuple[str, np.ndarray]:
    """'S02: buku, botol, 2 gunting' -> ('S02', vektor 8 jumlah). Gagal keras bila format aneh."""
    id_susunan, _, daftar = str(teks_susunan).partition(":")
    id_susunan = id_susunan.strip().upper()
    if not re.fullmatch(r"S\d{2}", id_susunan) or not daftar.strip():
        raise ValueError(f"format susunan tidak dikenal: {teks_susunan!r}")

    target = np.zeros(len(KELAS_OBJEK), dtype=np.float32)
    for item in daftar.split(","):
        item = item.strip().lower()
        if not item:
            continue
        m = re.fullmatch(r"(\d+)\s+(.+)", item)
        jumlah, nama = (float(m.group(1)), m.group(2).strip()) if m else (1.0, item)
        if nama not in KELAS_OBJEK:
            raise ValueError(f"kelas {nama!r} tidak dikenal pada {teks_susunan!r}")
        target[KELAS_OBJEK.index(nama)] += jumlah

    for nama, jumlah in KOREKSI_LABEL.get(id_susunan, {}).items():
        target[KELAS_OBJEK.index(nama)] = jumlah
    return id_susunan, target


def _baca_citra(path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").crop(CROP_A3).resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
    return np.transpose(arr, (2, 0, 1))                  # [C, H, W] untuk PyTorch


def _baca_kamera_index(akar_raw) -> tuple[np.ndarray, np.ndarray]:
    path_idx = akar_raw / "kamera_index.csv"
    if path_idx.exists():
        df_cam = pd.read_csv(path_idx)
    else:
        bagian = sorted((akar_raw / "kamera_index").glob("bagian-*.csv"))
        df_cam = pd.concat([pd.read_csv(p) for p in bagian], ignore_index=True)
    df_cam = df_cam.sort_values("t_host").reset_index(drop=True)
    return df_cam["t_host"].to_numpy(dtype=float), df_cam["berkas"].astype(str).str.strip().to_numpy()


def muat():
    """Memuat citra kamera, mencocokkan dengan selang waktu kondisi, dan membagi data per susunan."""
    if CROP_A3 is None:
        raise RuntimeError("CROP_A3 belum diisi: tetapkan koordinat area A3 dari citra asli dulu")

    akar_raw = manifest.AKAR / "data" / "raw"
    df_label = pd.read_csv(akar_raw / SESI / "label.csv")
    t_cam, berkas_cam = _baca_kamera_index(akar_raw)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    grup_susunan: list[str] = []
    n_hilang, kondisi_kosong = 0, []

    for nomor, (_, row) in enumerate(df_label.iterrows(), start=1):
        if nomor in KONDISI_DIBUANG:
            continue
        id_susunan, y_vec = _parse_susunan(row["susunan"])
        if id_susunan in SUSUNAN_DIBUANG:
            continue

        t_mulai = float(row["t_mulai_host"]) + 1.0  # buang 1 detik pertama tiap kondisi
        t_selesai = float(row["t_selesai_host"])
        n_awal = len(X_list)
        for nama_berkas in berkas_cam[(t_cam >= t_mulai) & (t_cam <= t_selesai)]:
            path_img = akar_raw / "kamera" / nama_berkas
            if not path_img.exists():
                n_hilang += 1
                continue
            X_list.append(_baca_citra(path_img))
            y_list.append(y_vec)
            grup_susunan.append(id_susunan)
        if len(X_list) == n_awal:
            kondisi_kosong.append(nomor)

    if not X_list:
        raise RuntimeError("tidak ada citra terbaca: periksa kolom 'berkas' di kamera_index.csv dan folder kamera/")
    if n_hilang or kondisi_kosong:
        print(f"[data] peringatan: {n_hilang} berkas citra tidak ditemukan; "
              f"kondisi tanpa citra: {kondisi_kosong}", file=sys.stderr)

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    grup = np.array(grup_susunan)

    # Lipatan per susunan. Permutasi ini sama dengan versi sebelumnya, jadi lipatan 0
    # berisi 10 susunan yang sama dengan set validasi lama.
    susunan_unik = np.sort(np.unique(grup))
    susunan_acak = np.random.default_rng(split.SEED_SPLIT).permutation(susunan_unik)
    lipatan = np.array_split(susunan_acak, JUMLAH_LIPATAN)
    k = int(os.environ.get("LIPATAN", "0"))
    susunan_uji = lipatan[k]

    # Validasi diambil dari susunan non-uji memakai split beku starter (seed 2026)
    susunan_sisa = np.sort(np.setdiff1d(susunan_unik, susunan_uji))
    idx_latih, idx_val = split.indeks(len(susunan_sisa), porsi_val=PORSI_VAL)
    susunan_latih, susunan_val = susunan_sisa[idx_latih], susunan_sisa[idx_val]

    m_latih = np.isin(grup, susunan_latih)
    m_val = np.isin(grup, susunan_val)
    m_uji = np.isin(grup, susunan_uji)
    return X[m_latih], y[m_latih], X[m_val], y[m_val], X[m_uji], y[m_uji]