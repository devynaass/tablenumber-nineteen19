"""Jalur CNN dari nol (Modul 05) - Tim 19: hitung objek di meja (regresi 8 kelas).

Jalankan dari akar repo:
    python models/123450007/run.py --seed 1

Kerangka ini belum diuji (belum dijalankan di laptop). Titik yang perlu disesuaikan
ditandai "SESUAIKAN".
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch import nn

JUMLAH_KELAS = 8  # buku, botol, ponsel, gunting, spidol, gelas, kotak pensil, kunci


# ----------------------------------------------------------------------------- data
def data_dummy(n: int = 60):
    """Dipakai hanya bila muat() belum diisi. Bentuk sama dengan data meja."""
    rng = np.random.default_rng(0)
    X = rng.random((n, 3, 224, 224), dtype=np.float32)
    y = rng.integers(0, 3, size=(n, JUMLAH_KELAS)).astype(np.float32)
    a, b = int(n * 0.6), int(n * 0.8)
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


def ambil_data():
    try:
        from src.protokol.data import muat

        return muat(), False
    except NotImplementedError:
        print("PERINGATAN: src/protokol/data.py belum diisi. Memakai data DUMMY, "
              "hasilnya hanya untuk memastikan kode berjalan.")
        return data_dummy(), True


def ke_nchw(X: np.ndarray) -> np.ndarray:
    """Pastikan bentuk [N, C, 224, 224] bertipe float32."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:                      # [N, H, W] -> tambah kanal
        X = X[:, None, :, :]
    elif X.ndim == 4 and X.shape[-1] in (1, 3) and X.shape[1] not in (1, 3):
        X = np.transpose(X, (0, 3, 1, 2))  # [N, H, W, C] -> [N, C, H, W]
    return X


# ----------------------------------------------------------------------------- model
def blok(masuk: int, keluar: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(masuk, keluar, kernel_size=3, padding=1),
        nn.BatchNorm2d(keluar),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNHitung(nn.Module):
    """Masukan [N, C, 224, 224] -> keluaran [N, 8] (jumlah per kelas, tanpa softmax)."""

    def __init__(self, kanal: int = 3):
        super().__init__()
        self.fitur = nn.Sequential(blok(kanal, 16), blok(16, 32), blok(32, 64), blok(64, 128))
        self.rata = nn.AdaptiveAvgPool2d(1)
        self.kepala = nn.Linear(128, JUMLAH_KELAS)

    def forward(self, x):
        x = self.rata(self.fitur(x)).flatten(1)
        return self.kepala(x)


# ----------------------------------------------------------------------------- latih
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch(X, y, ukuran, acak):
    idx = np.random.permutation(len(X)) if acak else np.arange(len(X))
    for i in range(0, len(X), ukuran):
        j = idx[i:i + ukuran]
        yield torch.from_numpy(X[j]), torch.from_numpy(y[j])


@torch.no_grad()
def prediksi(model, X, ukuran=16):
    model.eval()
    return np.concatenate([model(xb).numpy() for xb, _ in batch(X, np.zeros((len(X), JUMLAH_KELAS), np.float32), ukuran, False)])


def latih(model, X, y, epoch, lr=1e-3, ukuran=16):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    for e in range(epoch):
        model.train()
        total = 0.0
        for xb, yb in batch(X, y, ukuran, True):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch {e + 1}/{epoch}  loss {total / len(X):.4f}")


# ----------------------------------------------------------------------------- hasil
def ringkas(y_benar, y_pred):
    """MAE per kelas dan ketepatan susunan (semua 8 hitungan benar setelah dibulatkan)."""
    mae_kelas = np.abs(y_pred - y_benar).mean(axis=0)
    tepat = float((np.rint(y_pred).clip(0) == y_benar).all(axis=1).mean())
    return mae_kelas, tepat


def serahkan_ke_evaluate(model, y_benar, y_pred, seed):
    """SESUAIKAN: tulis berkas hasil lewat src.common.evaluate.

    Buka src/common/evaluate.py, lihat nama fungsi dan argumennya, lalu panggil di sini
    (biasanya metrics.json: MAE per kelas, ketepatan susunan, ukuran ONNX, latensi).
    """
    from src.common import evaluate  # noqa: F401

    raise NotImplementedError("Isi pemanggilan evaluate sesuai antarmuka di src/common/evaluate.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=2)
    a = ap.parse_args()
    set_seed(a.seed)

    (Xtr, ytr, Xva, yva, Xte, yte), dummy = ambil_data()
    Xtr, Xva = ke_nchw(Xtr), ke_nchw(Xva)
    ytr, yva = np.asarray(ytr, np.float32), np.asarray(yva, np.float32)
    assert Xtr.shape[2:] == (224, 224), f"masukan harus 224x224, dapat {Xtr.shape}"
    assert ytr.shape[1] == JUMLAH_KELAS, f"y harus [N, 8], dapat {ytr.shape}"

    model = CNNHitung(kanal=Xtr.shape[1])
    latih(model, Xtr, ytr, a.epoch)

    pred = prediksi(model, Xva)
    assert pred.shape == yva.shape, (pred.shape, yva.shape)
    mae_kelas, tepat = ringkas(yva, pred)
    print("MAE per kelas:", np.round(mae_kelas, 3))
    print("Ketepatan susunan:", round(tepat, 3))

    if dummy:
        print("Selesai (data dummy). Hasil tidak ditulis ke metrics.json.")
        return
    serahkan_ke_evaluate(model, yva, pred, a.seed)


if __name__ == "__main__":
    main()