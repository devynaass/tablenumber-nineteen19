"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from PIL import Image

from src.common import manifest

TUGAS = "regresi"

# Urutan baku 8 kelas objek pada meja
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


def _parse_susunan(teks_susunan: str) -> tuple[str, np.ndarray]:
    """Mengekstrak ID susunan (mis. 'S16') dan vektor jumlah 8 kelas objek."""
    bagian = str(teks_susunan).split(":", 1)
    id_susunan = bagian[0].strip()
    target = np.zeros(len(KELAS_OBJEK), dtype=np.float32)

    if len(bagian) > 1:
        for item in bagian[1].split(","):
            item = item.strip().lower()
            if not item:
                continue
            m = re.match(r"^(\d+)\s+(.+)$", item)
            if m:
                jumlah = float(m.group(1))
                nama = m.group(2).strip()
            else:
                jumlah = 1.0
                nama = item

            if nama in KELAS_OBJEK:
                target[KELAS_OBJEK.index(nama)] = jumlah

    return id_susunan, target


def muat():
    """Memuat dataset citra kamera, menyinkronkan label waktu, dan membagi fold berdasarkan ID susunan."""
    akar_raw = manifest.AKAR / "data" / "raw"
    path_label = akar_raw / "sesi-20261116-165743" / "label.csv"
    path_kamera_idx = akar_raw / "kamera_index.csv"
    dir_kamera = akar_raw / "kamera"

    df_label = pd.read_csv(path_label)

    # Baca kamera_index.csv utama, atau gabungkan dari folder kamera_index/ jika kosong
    if path_kamera_idx.exists():
        df_cam = pd.read_csv(path_kamera_idx)
    else:
        berkas_bagian = sorted((akar_raw / "kamera_index").glob("bagian-*.csv"))
        df_cam = pd.concat([pd.read_csv(p) for p in berkas_bagian], ignore_index=True)

    df_cam = df_cam.sort_values("t_host").reset_index(drop=True)
    t_cam = df_cam["t_host"].to_numpy()
    berkas_cam = df_cam["berkas"].to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    grup_susunan: list[str] = []

    for _, row in df_label.iterrows():
        t_mulai = float(row["t_mulai_host"]) + 1.0  # Buang 1 detik pertama per kondisi
        t_selesai = float(row["t_selesai_host"])
        id_susunan, y_vec = _parse_susunan(row["susunan"])

        # Ambil frame kamera di dalam rentang waktu kondisi
        mask = (t_cam >= t_mulai) & (t_cam <= t_selesai)
        for nama_berkas in berkas_cam[mask]:
            path_img = dir_kamera / str(nama_berkas).strip()
            if not path_img.exists():
                continue

            with Image.open(path_img) as img:
                img = img.convert("RGB")
                # TODO: Sesuaikan koordinat crop (left, top, right, bottom) area meja A3 jika diperlukan
                # w, h = img.size
                # img = img.crop((left, top, right, bottom))
                img = img.resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
                arr = np.transpose(arr, (2, 0, 1))  # Ubah ke [C, H, W] untuk PyTorch

            X_list.append(arr)
            y_list.append(y_vec)
            grup_susunan.append(id_susunan)

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    grup = np.array(grup_susunan)

    # Pembagian data deterministik (seed 2026) berdasarkan ID susunan (30 susunan -> 3 fold @ 10 susunan)
    susunan_unik = np.sort(np.unique(grup))
    rng = np.random.default_rng(2026)
    susunan_acak = rng.permutation(susunan_unik)

    # Fold 0 (10 susunan pertama) sebagai validasi, sisanya (20 susunan) sebagai latih
    susunan_val = set(susunan_acak[:10])
    mask_val = np.isin(grup, list(susunan_val))
    mask_train = ~mask_val

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]

    return X_train, y_train, X_val, y_val, None, None