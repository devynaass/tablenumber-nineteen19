"""models/123450085/run.py  –  CNN-LSTM (Jalur RNN), Sprint 1.

    python models/123450085/run.py --seed 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Path setup ────────────────────────────────────────────────────────────────
AKAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, AKAR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CNNLSTM                       # model.py kamu (tidak diubah)
from src.protokol import data as protokol_data  # protokol tim (dibekukan)

# ── Konstanta ─────────────────────────────────────────────────────────────────
NIM     = "123450085"
JALUR   = "rnn"
N_KELAS = 8      # jumlah kelas objek (sesuai KELAS_OBJEK di data.py)
UKURAN  = 224    # resolusi input model


def ke_tensor_X(X: np.ndarray, batch: int = 32) -> torch.Tensor:
    """Pastikan array X jadi (N, C, 224, 224) float32 dalam [0,1]."""
    t = torch.as_tensor(np.asarray(X, dtype=np.float32))

    # Normalkan dimensi jika belum (N,C,H,W)
    if t.ndim == 2:
        s = int(round(t.shape[1] ** 0.5))
        t = t.reshape(len(t), 1, s, s)
    elif t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
        t = t.permute(0, 3, 1, 2)   # (N,H,W,C) -> (N,C,H,W)

    if t.max() > 1.5:
        t = t / 255.0

    if t.shape[-2] != UKURAN or t.shape[-1] != UKURAN:
        t = torch.nn.functional.interpolate(
            t, size=(UKURAN, UKURAN), mode="bilinear", align_corners=False
        )
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="CNN-LSTM Jalur RNN – Sprint 1")
    ap.add_argument("--seed",       type=int,   default=1)
    ap.add_argument("--epochs",     type=int,   default=5)
    ap.add_argument("--batch-size", type=int,   default=8)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--perangkat",  default="cpu laptop")
    a = ap.parse_args()

    # 1. Seed ──────────────────────────────────────────────────────────────────
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    print("=" * 60)
    print("   CNN-LSTM (JALUR RNN) – Sprint 1 – Seed", a.seed)
    print("=" * 60)

    # 2. Muat data primer via protokol tim ────────────────────────────────────
    print("[1/4] Memuat data via src.protokol.data.muat() ...")
    X_tr_raw, y_tr_raw, X_va_raw, y_va_raw, _, _ = protokol_data.muat()

    X_tr = np.asarray(X_tr_raw, dtype=np.float32)
    y_tr = np.asarray(y_tr_raw, dtype=np.float32)   # (N, 8) regresi
    X_va = np.asarray(X_va_raw, dtype=np.float32)
    y_va = np.asarray(y_va_raw, dtype=np.float32)

    print(f"   Train : X={X_tr.shape}  y={y_tr.shape}")
    print(f"   Val   : X={X_va.shape}  y={y_va.shape}")

    X_tr_t = ke_tensor_X(X_tr)
    X_va_t = ke_tensor_X(X_va)
    y_tr_t = torch.as_tensor(y_tr)   # (N, 8) float

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=a.batch_size,
        shuffle=True,
    )

    # 3. Inisialisasi model ───────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CNNLSTM dari model.py kamu – outputnya N_KELAS = 8 (regresi)
    model   = CNNLSTM(num_classes=N_KELAS).to(device)
    crit    = nn.MSELoss()   # tugas REGRESI: MSE, bukan CrossEntropy
    opt     = torch.optim.Adam(model.parameters(), lr=a.lr)

    # 4. Latih ────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Melatih pada {device}  ({a.epochs} epoch, batch {a.batch_size}) ...")
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"   Epoch [{ep+1}/{a.epochs}]  MSE-Loss: {total/len(train_loader):.4f}")

    durasi = time.time() - t0
    print(f"   Selesai dalam {durasi:.1f} detik.")

    # 5. Inferensi validasi ───────────────────────────────────────────────────
    print("[3/4] Inferensi data validasi ...")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_va_t), a.batch_size):
            out = model(X_va_t[i : i + a.batch_size].to(device))
            preds.append(out.cpu().numpy())
    p_val = np.clip(np.concatenate(preds), 0, None)   # jumlah ≥ 0

    # 6. Simpan pred_val.npy & panggil evaluate resmi tim ────────────────────
    print("[4/4] Simpan prediksi & panggil src.common.evaluate ...")
    out_dir   = os.path.join(AKAR, "results", NIM)
    os.makedirs(out_dir, exist_ok=True)
    pred_file = os.path.join(out_dir, "pred_val.npy")
    np.save(pred_file, p_val)

    subprocess.run(
        [
            sys.executable, "-m", "src.common.evaluate",
            "--nim",       NIM,
            "--model",     JALUR,
            "--seed",      str(a.seed),
            "--pred-val",  pred_file,
            "--durasi",    f"{durasi:.1f}",
            "--perangkat", a.perangkat,
        ],
        check=True,
        cwd=AKAR,
    )

    print("\n" + "=" * 60)
    print("SUKSES! Cek results/123450085/metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
