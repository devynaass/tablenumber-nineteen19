"""Jalur CNN (Modul 05): CNN kecil dari nol untuk menaksir jumlah 8 kelas objek di meja.

    python models/<nim>/run.py --seed 1            # latih + prediksi validasi
    python models/<nim>/run.py --seed 1 --final    # sprint terakhir saja, oleh Penjaga Data

Nilai bawaan argumen adalah konfigurasi yang dipanggil run_all.py. Untuk tuning dan ablation:
    --lr 3e-4   --lebar 16   --dropout 0   --loss l1   --tanpa-aug   --epoch 60
Setiap run menambah satu baris ke models/<nim>/riwayat_latih.csv (bahan catatan.md).
Setiap run juga menimpa results/<nim>/metrics.json, jadi yang di-commit harus hasil
konfigurasi akhir.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

FOLDER = Path(__file__).resolve().parent
NIM = FOLDER.name        # nama folder = NIM
JALUR = "cnn"


# ---------------------------------------------------------------- utilitas
def atur_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ke_nchw(X) -> np.ndarray:
    """Pastikan bentuk [N, 3, H, W] float32."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] != 3 and X.shape[-1] == 3:
        X = X.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(X)


def nama_kelas(n: int) -> list[str]:
    return list(getattr(data, "KELAS_OBJEK", [f"kelas{i}" for i in range(n)]))


def metrik_kasus(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    """MAE per kelas dan ketepatan susunan (prediksi dibulatkan, minimal 0, harus sama persis 8 kelas)."""
    mae_kelas = np.abs(p - y).mean(axis=0)
    bulat = np.clip(np.rint(p), 0, None)
    return mae_kelas, float((bulat == y).all(axis=1).mean())


# ---------------------------------------------------------------- model
def buat_model(lebar: int, dropout: float, n_keluaran: int):
    import torch.nn as nn

    def konv(ci: int, co: int, stride: int = 1) -> list:
        return [nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]

    w = lebar
    return nn.Sequential(
        *konv(3, w, stride=2),                                   # 224 -> 112
        *konv(w, w), nn.MaxPool2d(2),                            # -> 56
        *konv(w, 2 * w), *konv(2 * w, 2 * w), nn.MaxPool2d(2),   # -> 28
        *konv(2 * w, 4 * w), *konv(4 * w, 4 * w), nn.MaxPool2d(2),  # -> 14
        *konv(4 * w, 4 * w),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(4 * w, 64), nn.ReLU(inplace=True),
        nn.Linear(64, n_keluaran), nn.Softplus(),                # jumlah objek tidak pernah negatif
    )


def augmentasi(xb, gen, geser: int):
    """Hanya untuk data latih. Kecerahan/kontras (meniru lampu 0 dan 255) dan geser kecil. Tanpa flip."""
    import torch
    import torch.nn.functional as F

    B, _, H, W = xb.shape
    kontras = torch.empty(B, 1, 1, 1).uniform_(0.7, 1.3, generator=gen)
    terang = torch.empty(B, 1, 1, 1).uniform_(-0.1, 0.1, generator=gen)
    xb = (xb * kontras + terang).clamp(0, 1)
    if geser > 0:
        xp = F.pad(xb, (geser, geser, geser, geser), mode="replicate")
        dx = torch.randint(0, 2 * geser + 1, (B,), generator=gen)
        dy = torch.randint(0, 2 * geser + 1, (B,), generator=gen)
        xb = torch.stack([xp[i, :, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])
    return xb


def prediksi(model, X: np.ndarray, rerata, simpangan, perangkat, batch: int = 64) -> np.ndarray:
    import torch

    model.eval()
    keluar = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = (torch.from_numpy(X[i:i + batch]) - rerata) / simpangan
            keluar.append(model(xb.to(perangkat)).cpu().numpy())
    return np.concatenate(keluar).astype(np.float32)


# ---------------------------------------------------------------- latih
def latih_dan_prediksi(seed: int, final: bool, a: argparse.Namespace):
    import torch

    atur_seed(seed)
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    Xtr, Xva = ke_nchw(Xtr), ke_nchw(Xva)
    Xte = ke_nchw(Xte) if final and Xte is not None else None
    ytr, yva = np.asarray(ytr, np.float32), np.asarray(yva, np.float32)
    print(f"[cnn] latih {Xtr.shape}, val {Xva.shape}, perangkat {perangkat}", flush=True)

    # normalisasi per kanal dari data latih saja
    rerata = torch.from_numpy(Xtr.mean(axis=(0, 2, 3), keepdims=True))
    simpangan = torch.from_numpy(Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6)

    model = buat_model(a.lebar, a.dropout, ytr.shape[1]).to(perangkat)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    jadwal = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epoch)
    fungsi_loss = torch.nn.L1Loss() if a.loss == "l1" else torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    mae_terbaik, bobot_terbaik, epoch_terbaik, sabar = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(1, a.epoch + 1):
        model.train()
        urut = torch.randperm(len(Xtr_t), generator=gen)
        total = 0.0
        for i in range(0, len(urut), a.batch):
            idx = urut[i:i + a.batch]
            xb = Xtr_t[idx] if a.tanpa_aug else augmentasi(Xtr_t[idx], gen, a.geser)
            xb = ((xb - rerata) / simpangan).to(perangkat)
            yb = ytr_t[idx].to(perangkat)
            opt.zero_grad()
            loss = fungsi_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        jadwal.step()

        mae_val = float(np.abs(prediksi(model, Xva, rerata, simpangan, perangkat) - yva).mean())
        print(f"epoch {ep:3d}  loss {total / len(urut):.4f}  val_mae {mae_val:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if mae_val < mae_terbaik:
            mae_terbaik, bobot_terbaik, epoch_terbaik, sabar = mae_val, copy.deepcopy(model.state_dict()), ep, 0
        else:
            sabar += 1
        if sabar >= a.sabar:
            print(f"[cnn] berhenti dini: {a.sabar} epoch tanpa perbaikan", flush=True)
            break
        if time.time() - t0 > a.maks_menit * 60:
            print(f"[cnn] anggaran {a.maks_menit} menit habis", flush=True)
            break

    model.load_state_dict(bobot_terbaik)
    torch.save(model.state_dict(), FOLDER / f"cnn_seed{seed}.pt")   # *.pt tidak di-commit (.gitignore)
    pred_val = prediksi(model, Xva, rerata, simpangan, perangkat)
    pred_test = prediksi(model, Xte, rerata, simpangan, perangkat) if Xte is not None else None
    return pred_val, pred_test, yva, epoch_terbaik


def catat_riwayat(a, seed, epoch_terbaik, mae_kelas, tepat, durasi, perangkat) -> None:
    berkas = FOLDER / "riwayat_latih.csv"
    kelas = nama_kelas(len(mae_kelas))
    baris = {"waktu": datetime.now().strftime("%Y-%m-%d %H:%M"), "seed": seed, "lr": a.lr, "wd": a.wd,
             "lebar": a.lebar, "dropout": a.dropout, "loss": a.loss, "aug": not a.tanpa_aug,
             "epoch_maks": a.epoch, "epoch_terbaik": epoch_terbaik,
             "val_mae": round(float(mae_kelas.mean()), 4), "ketepatan_susunan": round(tepat, 4),
             **{f"mae_{k.replace(' ', '_')}": round(float(v), 4) for k, v in zip(kelas, mae_kelas)},
             "durasi_detik": round(durasi, 1), "perangkat": perangkat}
    baru = not berkas.exists()
    with berkas.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris))
        if baru:
            w.writeheader()
        w.writerow(baris)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default=None, help="bawaan: terdeteksi otomatis")
    ap.add_argument("--epoch", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lebar", type=int, default=32, help="jumlah filter blok pertama")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--loss", choices=["l1", "mse"], default="l1")
    ap.add_argument("--tanpa-aug", action="store_true")
    ap.add_argument("--geser", type=int, default=8, help="geser acak maksimum (piksel)")
    ap.add_argument("--sabar", type=int, default=15, help="epoch tanpa perbaikan sebelum berhenti dini")
    ap.add_argument("--maks-menit", type=float, default=18.0)
    a = ap.parse_args()

    import torch

    perangkat = a.perangkat or (f"cuda {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu laptop")
    t0 = time.time()
    pred_val, pred_test, yva, epoch_terbaik = latih_dan_prediksi(a.seed, a.final, a)
    durasi = time.time() - t0

    mae_kelas, tepat = metrik_kasus(yva, pred_val)
    print("\nMAE validasi per kelas:")
    for k, v in zip(nama_kelas(len(mae_kelas)), mae_kelas):
        print(f"  {k:13s} {v:.4f}")
    print(f"ketepatan susunan (val): {tepat:.4f}   epoch terbaik: {epoch_terbaik}\n")
    catat_riwayat(a, a.seed, epoch_terbaik, mae_kelas, tepat, durasi, perangkat)

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()